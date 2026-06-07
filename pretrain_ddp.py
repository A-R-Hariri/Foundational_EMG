"""
DDP pretraining: VICReg on fixed-shape windows from a single window size.

Uses the same data_pickles/ directory as pretrain_fsdp.py but filters to one
window size (TARGET_WIN_SEC) for fixed-shape batching without custom collation.

Usage:
    torchrun --nproc_per_node=<N> pretrain_ddp.py
    python pretrain_ddp.py  # single GPU
"""

import os
import warnings
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.multiprocessing as mp
from tqdm import tqdm

from models import EMGTransformer
from utils import (
    DATA_PATH, CKPT_PATH, DDP_BATCH_SIZE, EPOCHS, LR, MIN_LR, LR_FACTOR,
    LR_PATIENCE, PATIENCE, TARGET_WIN_SEC, SEQ, SEED,
    vicreg_loss, augment_gpu, sync_mean,
    create_ddp_loaders, count_params, save_checkpoint, load_checkpoint,
)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_DEBUG", "WARN")
warnings.filterwarnings("ignore")

mp.set_sharing_strategy("file_system")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -------- model config --------
D_MODEL = 128
N_HEADS = 2
N_LAYERS = 4


def run_epoch(epoch, model, loader, optimizer, scaler, device, train=True):
    model.train(train)
    loss_sum, count = 0.0, 0
    etype = "Train" if train else "Val"
    rank = dist.get_rank() if dist.is_initialized() else 0
    pbar = tqdm(
        desc=f"{etype} Ep {epoch}", total=len(loader),
        leave=False, dynamic_ncols=True,
        bar_format="{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]",
        mininterval=1, disable=(rank != 0),
    )
    for x in loader:
        x = x.to(device, non_blocking=True)
        x1, x2 = augment_gpu(x), augment_gpu(x)
        if train:
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda"):
                loss = vicreg_loss(model(x1), model(x2))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad(), autocast(device_type="cuda"):
                loss = vicreg_loss(model(x1), model(x2))
        loss_sum += loss.detach().item()
        count += 1
        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss_sum / count:.4f}"})
    pbar.close()
    return loss_sum / max(1, count)


def main():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        distributed = True
    else:
        rank = world = 1
        local_rank = 0
        distributed = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(CKPT_PATH, exist_ok=True)

    train_loader, val_loader, train_sampler, _ = create_ddp_loaders(
        DATA_PATH, DDP_BATCH_SIZE, ws_filter=TARGET_WIN_SEC, world_size=world, rank=rank
    )

    model = EMGTransformer(
        d_model=D_MODEL, nhead=N_HEADS, num_layers=N_LAYERS, max_len=SEQ,
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False, static_graph=True)

    if rank == 0:
        m = model.module if distributed else model
        print(f"Parameters: {count_params(m):,}")

    opt = Adam(model.parameters(), lr=LR)
    sch = ReduceLROnPlateau(opt, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR)
    scaler = GradScaler()

    ckpt_best = os.path.join(CKPT_PATH, "ddp_best.pt")
    ckpt_latest = os.path.join(CKPT_PATH, "ddp_latest.pt")
    start_epoch, best, wait = 0, float("inf"), 0

    if os.path.exists(ckpt_best):
        try:
            start_epoch, best = load_checkpoint(model, ckpt_best, rank, opt, sch, scaler)
        except Exception as e:
            if rank == 0:
                print(f"Could not load checkpoint: {e}")

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = run_epoch(epoch, model, train_loader, opt, scaler, device, train=True)
        val_loss = run_epoch(epoch, model, val_loader, None, None, device, train=False)

        if distributed:
            train_loss = sync_mean(train_loss, device)
            val_loss = sync_mean(val_loss, device)

        if rank == 0:
            print(f"Ep {epoch:3d} | train={train_loss:.4f} val={val_loss:.4f} lr={opt.param_groups[0]['lr']:.2e}")

        sch.step(val_loss)

        if rank == 0:
            save_checkpoint(model, ckpt_latest, epoch=epoch, optimizer=opt,
                            scheduler=sch, scaler=scaler, best_val=best)
            if val_loss < best:
                best = val_loss
                save_checkpoint(model, ckpt_best, epoch=epoch, optimizer=opt,
                                scheduler=sch, scaler=scaler, best_val=best)
                print(f"  New best: {best:.4f}")
                wait = 0
            else:
                wait += 1

        if wait >= PATIENCE:
            if rank == 0:
                print("Early stopping.")
            break

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
