import os
os.environ.setdefault("OMP_NUM_THREADS", "1")                      
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")      
os.environ.setdefault("NCCL_DEBUG", "WARN")                        

import random, warnings, gc
warnings.filterwarnings("ignore")

import torch
torch.backends.cudnn.benchmark = True                              
torch.backends.cuda.matmul.allow_tf32 = True                       
torch.backends.cudnn.allow_tf32 = True                             

import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")                             
print(torch.cuda.is_available(), torch.cuda.device_count())

import numpy as np
from tqdm import tqdm

# ----------------- Config -----------------
PATH = "pickles/"
TARGET_FS = 500
WIN_RANGE = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
MAX_WIN_SAMPLES = int(WIN_RANGE[-1] * TARGET_FS)

TARGET_CH   = 8
BATCH_SIZE  = 2048   
LATENT_DIM  = 128
NUM_HEADS   = 2
NUM_BLOCKS  = 4
DROPOUT     = 0.1
EPOCHS      = 200
LR          = 1e-3
MIN_LR      = 1e-5
PATIENCE    = 5         
RL_FACT     = 0.8
RL_PATIENCE = 1

SEED = 42
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

# ----------------- Dataset -----------------
class EMGDataset(Dataset):
    """
    Expects many files like X_train_*.npy / X_val_*.npy with shape (N, 200, 8) float16/float32.
    mmap_mode='r' keeps RAM usage minimal. Index maps (file_id, row_id) over all files.
    """
    def __init__(self, path, split="train"):
        self.files = sorted(
            os.path.join(path, f) for f in os.listdir(path)
            if f.startswith(f"X_{split}_") and f.endswith(".npy")
        )
        if not self.files:
            raise FileNotFoundError(f"No npy files found for split='{split}' in {path}")
        self.maps = [np.load(fp, mmap_mode="r") for fp in self.files]
        self.index = [(fi, i) for fi, m in enumerate(self.maps) for i in range(m.shape[0])]
        random.shuffle(self.index)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, i = self.index[idx]
        # (T,C) -> torch.float32 on CPU; augmentations remain in torch
        x = torch.from_numpy(self.maps[fi][i])  # (L=200, C=8)
        # return self.augment(x), self.augment(x)
        return x
    
    # @staticmethod
    # def get_tail_mask(x: torch.Tensor) -> torch.Tensor:
    #     # x: (L, C)
    #     is_pad = (x.abs().sum(dim=-1) == 0)          # True where row is all zeros
    #     nonpad = ~is_pad                             # (L,)
    #     L = x.size(0)
    #     last_valid = nonpad.size(0) - 1 - nonpad.flip(dims=[0]).float().argmax()
    #     ar = torch.arange(L, device=x.device)
    #     mask_pad = ar > last_valid                   # True where padding (trailing)
    #     return mask_pad  

    # @staticmethod
    # def augment(x: torch.Tensor) -> torch.Tensor:
    #     # x: (L, C) float32
    #     y = x.clone()
    #     # (1) amplitude scaling
    #     if torch.rand(()).item() < 0.5:
    #         scale = torch.rand((), dtype=y.dtype).item() * 0.1 + 1.0
    #         y = y * scale
    #     # (2) temporal roll
    #     if torch.rand(()).item() < 0.5:                                               #    ----->      CPU chokes
    #         # limit shift a bit; using +/- 4 samples matches your earlier choice
    #         shift = torch.randint(-4, 4, ()).item()
    #         if shift:
    #             y = torch.roll(y, shifts=shift, dims=0)
    #     # (3) gaussian noise
    #     if torch.rand(()).item() < 0.5:
    #         std = y.std()
    #         if torch.isfinite(std) and std > 0:
    #             y = y + torch.randn_like(y) * (0.02 * std)
    #     # (4) channel dropout
    #     if torch.rand(()).item() < 0.1:
    #         num_ch = y.shape[1]
    #         drop_k = torch.randint(1, min(2, num_ch), ()).item()
    #         idx = torch.randperm(num_ch)[:drop_k]
    #         y[:, idx] = 0
    #     return y

# ----------------- Model -----------------
class PositionalEmbedding(nn.Module):
    """Learned position embedding for a fixed max length."""
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, mean=0.0, std=0.02)

    def forward(self, x):
        # x: (B, L, D)
        return x + self.pe[:, :x.size(1), :]

class EMGTransformer(nn.Module):
    def __init__(self, in_ch=TARGET_CH, d_model=LATENT_DIM, nhead=NUM_HEADS,
                 num_layers=NUM_BLOCKS, dropout=DROPOUT, max_len=MAX_WIN_SAMPLES):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 128, kernel_size=4, stride=1, padding='same')
        self.conv2 = nn.Conv1d(128, d_model, kernel_size=2, stride=1, padding='same')
        self.norm  = nn.LayerNorm(d_model)

        self.pos = PositionalEmbedding(max_len=max_len, d_model=d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=2*d_model, dropout=dropout,
            batch_first=True, norm_first=True  # better stability
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Linear(d_model, d_model)  # VICReg projection (identity dim)

    def forward(self, x, mask=None):
        # x: (B, L, C)
        x = self.trim_trailing_zeros(x)
        x = x.transpose(1, 2)                           # (B, C, L) for conv1d
        x = F.relu(self.conv1(x))               
        x = F.relu(self.conv2(x))                       # (B, D, L)
        x = x.transpose(1, 2)                           # (B, L, D)
        x = self.norm(x)
        x = self.pos(x)                                 # add learned positional embeddings
        x = self.encoder(x)                             # (B, L, D)
        x = x.mean(dim=1)                               # GAP over time -> (B, D)
        z = self.head(x)                                # (B, D)
        return z
    
    @staticmethod
    def trim_trailing_zeros(x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        # True where row is non-zero
        nonzero = (x.abs().sum(dim=-1) != 0)  
        flipped = nonzero.flip(dims=[1])
        last_valid = flipped.float().argmax(dim=1)  # distance from right edge
        valid_len = x.size(1) - last_valid          # actual valid length

        max_len = int(valid_len.max())              # tensor→int (GPU-safe)
        return x[:, :max_len, :]                    # trim trailing zeros

# ----------------- VICReg loss -----------------
def var_term(z, gamma):
    std = z.std(dim=0, unbiased=False)
    return torch.mean(F.relu(gamma - std))

def cov_term(z):
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (z.size(0) - 1)
    off = cov - torch.diag(torch.diag(cov))
    return (off ** 2).mean()

def vicreg_loss(z1, z2, lamb=25.0, mu=32.0, nu=1.0, gamma=1.0):
    sim = torch.mean((z1 - z2) ** 2)                  # Invariance
    v = var_term(z1, gamma) + var_term(z2, gamma)     # Variance
    c = cov_term(z1) + cov_term(z2)                   # Covariance
    return lamb * sim + mu * v + nu * c

# ----------------- Utils -----------------
def augment_gpu(x: torch.Tensor) -> torch.Tensor:
    """
    Apply the same random augmentation policy across the entire batch.
    Safe for DDP / multi-GPU use.
    x: (B, L, C) on GPU
    """
    y = x.clone()

    # (1) amplitude scaling
    if torch.rand((), device=x.device) < 0.5:
        scale = 1.0 + 0.1 * torch.randn((), device=x.device, dtype=x.dtype)
        y = y * scale

    # (2) temporal roll
    if torch.rand((), device=x.device) < 0.5:
        shift = int(torch.randint(-4, 5, (), device=x.device))
        y = torch.roll(y, shifts=shift, dims=2)  # roll along time (L)

    # (3) gaussian noise
    if torch.rand((), device=x.device) < 0.5:
        std = y.std()
        if torch.isfinite(std) and std > 0:
            y = y + 0.02 * std * torch.randn_like(y)

    # (4) channel dropout
    if torch.rand((), device=x.device) < 0.1:
        num_ch = y.shape[2]
        drop_k = int(torch.randint(1, min(2, num_ch), (), device=x.device))
        idx = torch.randperm(num_ch, device=x.device)[:drop_k]
        y[:, :, idx] = 0

    return y

def create_loaders(batch_size, world_size=1, rank=0):
    train_ds = EMGDataset(PATH, "train")
    val_ds = EMGDataset(PATH, "val")

    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        shuffle_train = False
        shuffle_val = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True
        shuffle_val = False

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train, sampler=train_sampler,
        pin_memory=True, persistent_workers=False, drop_last=True, 
        num_workers=8, prefetch_factor=16)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=shuffle_val, sampler=val_sampler,
        pin_memory=True, persistent_workers=False, drop_last=False, 
        num_workers=2, prefetch_factor=16)
    return train_loader, val_loader, train_sampler, val_sampler

# ----------------- Train/Eval -----------------
def run_epoch(epoch, model, loader, optimizer, scaler=None, device='cuda', train=True):
    model.train(mode=train)
    loss_sum, count = 0.0, 0
    etype = 'Train' if train else 'Val'
    pbar = tqdm(desc=f"{etype} - Ep: {epoch}", total=len(loader), leave=False, dynamic_ncols=True,
                bar_format='{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]', 
                mininterval=1, disable=not (not dist.is_initialized() or dist.get_rank() == 0))
    
    # for x1, x2 in loader:
    for x in loader:
        x = x.to(device, non_blocking=True)
        x1, x2 = augment_gpu(x), augment_gpu(x)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                with autocast(device_type="cuda"):
                    loss = vicreg_loss(model(x1), model(x2))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = vicreg_loss(model(x1), model(x2))
                loss.backward()
                optimizer.step()

        else:
            with torch.no_grad():
                if scaler:
                    with autocast(device_type="cuda"):
                        loss = vicreg_loss(model(x1), model(x2))
                else:
                    loss = vicreg_loss(model(x1), model(x2))

        loss_sum += loss.detach().item()
        count += 1
        if pbar.total is not None:
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss_sum / count:.4f}"})

        # gc.collect()
        # torch.cuda.empty_cache()

    if pbar.total is not None:
        pbar.close()

    return loss_sum / count

def save_checkpoint(model, path, epoch=None, optimizer=None, 
                    scheduler=None, scaler=None, best_val=None, save_optim=True):
    state = {
        "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
    }

    if save_optim and optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if save_optim and scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if save_optim and scaler is not None:
        state["scaler"] = scaler.state_dict()

    torch.save(state, path)


def load_checkpoint(model, path, rank, optimizer=None, scheduler=None, 
                    scaler=None, map_location="cuda", load_optim=True):
    ckpt = torch.load(path, map_location=map_location)

    if isinstance(model, DDP):
        model.module.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt["model"])

    if load_optim:
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

    if rank == 0:
        print(f"Loaded checkpoint: {path}")
        print(f"Epoch: {ckpt.get('epoch')}, Best Val: {ckpt.get('best_val')}")
        
    return ckpt.get("epoch", 0), ckpt.get("best_val", float('inf'))

# ----------------- Main (single-GPU or DDP) -----------------
def sync_mean(tensor, world_size, device):
    tensor = torch.tensor(tensor, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item() / world_size

NAME = "tr1"

def main():
    # Detect whether torchrun distributed mode or single GPU
    if "LOCAL_RANK" in os.environ:
        # ---------------- Distributed (torchrun) ----------------
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        distributed = True
    else:
        # ---------------- Single GPU / Python mode ----------------
        rank = 0
        world_size = 1
        local_rank = 0
        distributed = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data loaders
    train_loader, val_loader, train_sampler, val_sampler = create_loaders(BATCH_SIZE, world_size, rank)

    # Build model
    model = EMGTransformer(
        in_ch=TARGET_CH, d_model=LATENT_DIM, nhead=NUM_HEADS,
        num_layers=NUM_BLOCKS, dropout=DROPOUT, max_len=MAX_WIN_SAMPLES
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False, static_graph=True)

    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=RL_FACT, patience=RL_PATIENCE, min_lr=MIN_LR)
    scaler = GradScaler()

    if rank == 0:
        total_params = sum(p.numel() for p in (model.module.parameters() if distributed else model.parameters()))
        print(model, f"\nTotal parameters: {total_params:,}")

        print("Loading checkpoints...")
    try:
        start_epoch, best_val = load_checkpoint(
            model, f"{PATH + NAME}_best.pt", rank,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            load_optim=False)
    except FileNotFoundError:
        start_epoch, best_val = 0, float('inf')
        if rank == 0:
            print("No checkpoint found")
    wait = 0

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss = run_epoch(epoch, model, train_loader, optimizer, scaler=scaler, device=device, train=True)
        val_loss = run_epoch(epoch, model, val_loader, optimizer, scaler=scaler, device=device, train=False)

        if distributed:
            train_loss = sync_mean(train_loss, world_size, device)
            val_loss = sync_mean(val_loss, world_size, device)

        if rank == 0:
            print(f"train={train_loss:.4f}  val={val_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step(val_loss)

        if rank == 0:
            save_checkpoint(model, f"{PATH + NAME}_{epoch}.pt",
                epoch=epoch, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                best_val=best_val, save_optim=True)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            wait = 0
            if rank == 0:
                save_checkpoint(model, f"{PATH + NAME}_best.pt",
                    epoch=epoch, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    best_val=best_val, save_optim=True)
        else:
            wait += 1
            if wait >= PATIENCE and rank == 0:
                print(f"Early stopping at epoch {epoch}")
                break

    if rank == 0:
        print("Done.")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
