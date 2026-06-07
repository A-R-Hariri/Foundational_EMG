"""
FSDP pretraining: VICReg + MAE on variable-length, variable-channel cross-dataset EMG.

Usage:
    torchrun --nproc_per_node=<N> pretrain_fsdp.py
    python pretrain_fsdp.py  # single GPU
"""

import os
import math
import warnings
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision, BackwardPrefetch, ShardingStrategy,
    FullStateDictConfig, StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, apply_activation_checkpointing,
)
from tqdm import tqdm

from models import EMGPretrainer, Encoder, VICHead, MAEDecoder
from utils import (
    DATA_PATH, CKPT_PATH, BATCH_SIZE, EPOCHS, LR, MIN_LR, LR_FACTOR,
    LR_PATIENCE, PATIENCE, SEED, MAE_MASK_FRAC, MAE_LOSS_W, VIC_LOSS_W,
    vicreg_loss, generate_mae_mask, emg_augment,
    sync_mean, create_fsdp_loaders, count_params,
)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def run_epoch(epoch, model, loader, optimizer, scaler, device, train=True):
    model.train(train)
    total_loss = total_vic = total_mae = 0.0
    valid_steps = 0
    etype = "Train" if train else "Val"
    rank = dist.get_rank() if dist.is_initialized() else 0
    pbar = tqdm(
        total=len(loader), desc=f"{etype} Ep {epoch}",
        leave=False, dynamic_ncols=True,
        bar_format="{l_bar}{bar:20}{r_bar}", mininterval=1,
        disable=(rank != 0),
    )
    for step, (xs, time_masks, ch_masks, fs_list) in enumerate(loader):
        xs = xs.to(device, non_blocking=True).float()
        time_masks = time_masks.to(device, non_blocking=True)
        ch_masks = ch_masks.to(device, non_blocking=True)
        fs_list = fs_list.to(device, non_blocking=True)
        xs = torch.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            B, L, C = xs.shape
            t_idx = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0).expand(B, L)
            times_sec = t_idx / fs_list.unsqueeze(1).clamp_min(1e-6)
            x1 = emg_augment(xs, time_masks, ch_masks)
            x2 = emg_augment(xs, time_masks, ch_masks)
            with torch.no_grad():
                x1_norm = x1 / x1.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
            mae_mask = generate_mae_mask(time_masks).to(device)
            with autocast(device_type="cuda", enabled=True):
                z1_seq, m1 = model.forward_encoder(x1, time_masks, ch_masks, times_sec)
                z2_seq, m2 = model.forward_encoder(x2, time_masks, ch_masks, times_sec)
                loss_vic = vicreg_loss(model.project_vic(z1_seq, m1), model.project_vic(z2_seq, m2))
                z_mae, _ = model.forward_encoder(x1, time_masks, ch_masks, times_sec, mae_mask=mae_mask)
                recon = model.decode_mae(z_mae, L, C)
                valid_mask = time_masks.unsqueeze(-1) & ch_masks.unsqueeze(1)
                mae_pos = mae_mask.unsqueeze(-1) & valid_mask
                if mae_pos.any():
                    recon_norm = recon / recon.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
                    loss_mae = ((recon_norm - x1_norm)[mae_pos] ** 2).mean()
                else:
                    loss_mae = torch.zeros((), device=device)
                loss = VIC_LOSS_W * loss_vic + MAE_LOSS_W * loss_mae
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss={loss.item()}")
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.detach().item()
            total_vic += loss_vic.detach().item()
            total_mae += loss_mae.detach().item()
            valid_steps += 1
        except Exception as e:
            if rank == 0:
                pbar.write(f"Step {step} skipped: {e}")
            if train:
                optimizer.zero_grad(set_to_none=True)
        pbar.update(1)
        if rank == 0 and valid_steps > 0:
            pbar.set_postfix({
                "loss": f"{total_loss / valid_steps:.4f}",
                "vic": f"{total_vic / valid_steps:.4f}",
                "mae": f"{total_mae / valid_steps:.4f}",
            })
    pbar.close()
    return total_loss / max(1, valid_steps)


def _save_fsdp(model, opt, sch, scaler, path, epoch, best, global_max_C):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()
    if dist.get_rank() == 0:
        torch.save({
            "model": cpu_state,
            "optimizer": opt.state_dict(),
            "scheduler": sch.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch, "best": best,
            "global_max_C": global_max_C,
        }, path)


def _save_single(model, opt, sch, scaler, path, epoch, best, global_max_C):
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sch.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch, "best": best,
        "global_max_C": global_max_C,
    }, path)


def main():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
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

    train_loader, val_loader, train_sampler, global_max_C = create_fsdp_loaders(
        DATA_PATH, BATCH_SIZE, world, rank
    )
    if rank == 0:
        print(f"global_max_C={global_max_C}")

    model = EMGPretrainer(max_C=global_max_C)

    if distributed:
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=lambda m: isinstance(m, nn.TransformerEncoderLayer),
        )
        model = FSDP(
            model,
            auto_wrap_policy=ModuleWrapPolicy({Encoder, VICHead, MAEDecoder, nn.TransformerEncoderLayer}),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            mixed_precision=MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.float32,
            ),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
            limit_all_gathers=True,
        )
        dist.barrier()
    else:
        model = model.to(device)

    if rank == 0:
        print(f"Parameters: {count_params(model):,}")

    opt = Adam(model.parameters(), lr=LR)
    sch = ReduceLROnPlateau(opt, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR)
    scaler = GradScaler()

    ckpt_best = os.path.join(CKPT_PATH, "fsdp_best.pt")
    ckpt_latest = os.path.join(CKPT_PATH, "fsdp_latest.pt")
    best = float("inf")
    wait = 0

    if os.path.exists(ckpt_best):
        state = torch.load(ckpt_best, map_location="cpu")
        if distributed:
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                      FullStateDictConfig(offload_to_cpu=True, rank0_only=False)):
                model.load_state_dict(state["model"])
        else:
            model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sch.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        best = state.get("best", best)
        if rank == 0:
            print(f"Resumed from {ckpt_best} (best={best:.4f})")

    for epoch in range(1, EPOCHS + 1):
        if distributed:
            dist.barrier()
            train_sampler.set_epoch(epoch)

        train_loss = run_epoch(epoch, model, train_loader, opt, scaler, device, train=True)
        with torch.no_grad():
            val_loss = run_epoch(epoch, model, val_loader, None, None, device, train=False)

        if distributed:
            train_loss = sync_mean(train_loss, device)
            val_loss = sync_mean(val_loss, device)

        if rank == 0:
            print(f"Ep {epoch:3d} | train={train_loss:.4f} val={val_loss:.4f} lr={opt.param_groups[0]['lr']:.2e}")

        sch.step(val_loss)

        save = _save_fsdp if distributed else _save_single
        if not distributed or dist.get_rank() == 0 or distributed:
            if distributed:
                save(model, opt, sch, scaler, ckpt_latest, epoch, best, global_max_C)
            elif rank == 0:
                save(model, opt, sch, scaler, ckpt_latest, epoch, best, global_max_C)

        if val_loss < best:
            best = val_loss
            if distributed:
                save(model, opt, sch, scaler, ckpt_best, epoch, best, global_max_C)
            elif rank == 0:
                save(model, opt, sch, scaler, ckpt_best, epoch, best, global_max_C)
            if rank == 0:
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
