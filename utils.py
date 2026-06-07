import os
import gc
import re
import math
import glob
import bisect
import functools
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
import torch.distributed as dist
from tqdm import tqdm

# ======== CONFIG ========

DATA_PATH = "data_pickles"
CKPT_PATH = "checkpoints"
DTYPE = np.float16

# -------- training --------
BATCH_SIZE = 128
EPOCHS = 200
LR = 1e-4
MIN_LR = 1e-6
LR_FACTOR = 0.8
LR_PATIENCE = 2
PATIENCE = 10
SEED = 67

# -------- FSDP model --------
LATENT_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.1
CONV_KERNEL = 8
CONV_STRIDE = 4
CONV_PADDING = CONV_KERNEL // 2

# -------- SSL objectives --------
MAE_MASK_FRAC = 0.3
MAE_LOSS_W = 1.0
VIC_LOSS_W = 1.0

# -------- DDP fixed-shape --------
DDP_BATCH_SIZE = 2048
TARGET_WIN_SEC = 0.2
TARGET_FS = 500
SEQ = int(TARGET_WIN_SEC * TARGET_FS)

# -------- augmentation probs (variable-length) --------
AUG_PROBS = {
    "amp_global": 0.7,
    "amp_per_ch": 0.5,
    "baseline": 0.3,
    "tshift": 0.7,
    "time_warp": 0.5,
    "noise": 0.6,
    "ch_dropout": 0.2,
    "ch_perm": 0.5,
    "mag_warp": 0.5,
    "lowpass": 0.3,
}

# ======== MISC ========

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sync_mean(val, device):
    t = torch.tensor([float(val)], device=device, dtype=torch.float32)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t / dist.get_world_size()
    return float(t.item())

def _worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id + 1)
    torch.manual_seed(SEED + worker_id + 1)

# ======== LOSSES ========

def vicreg_loss(z1, z2, lamb=25.0, mu=25.0, nu=1.0, gamma=1.0):
    sim = ((z1 - z2) ** 2).mean()

    def var_term(z):
        return F.relu(gamma - z.std(dim=0, unbiased=False)).mean()

    def cov_term(z):
        zc = z - z.mean(dim=0)
        cov = (zc.T @ zc) / (zc.size(0) - 1 + 1e-6)
        off = cov - torch.diag(torch.diag(cov))
        return (off ** 2).mean()

    return lamb * sim + mu * (var_term(z1) + var_term(z2)) + nu * (cov_term(z1) + cov_term(z2))

def generate_mae_mask(time_mask, frac=MAE_MASK_FRAC):
    B, L = time_mask.shape
    mask = torch.zeros_like(time_mask)
    for b in range(B):
        valid = torch.where(time_mask[b])[0]
        k = int(max(1, frac * valid.numel()))
        if k > 0:
            sel = valid[torch.randperm(valid.numel(), device=time_mask.device)[:k]]
            mask[b, sel] = True
    return mask

# ======== AUGMENTATION ========

def _lowpass_avg(x, k=5):
    if k <= 1:
        return x
    csum = torch.cumsum(x, dim=1)
    ma = csum.clone()
    ma[:, k:, :] = csum[:, k:, :] - csum[:, :-k, :]
    ma = ma / k
    head_counts = torch.arange(1, k, device=x.device, dtype=x.dtype).view(1, -1, 1)
    ma[:, :k-1, :] = csum[:, :k-1, :] / head_counts
    return ma

def _magnitude_warp(x, prob=0.5, knots=4, amp=0.2):
    B, L, C = x.shape
    if torch.rand(()) >= prob or knots <= 1:
        return x
    scales = 1.0 + amp * (2 * torch.rand(B, knots, device=x.device) - 1.0)
    warp = F.interpolate(scales.unsqueeze(1), size=L, mode='linear', align_corners=True)
    return x * warp.squeeze(1).unsqueeze(-1)

def _time_warp(x, prob=0.5, factor_range=(0.95, 1.05)):
    B, L, C = x.shape
    if torch.rand(()) >= prob:
        return x
    factors = torch.empty(B, device=x.device).uniform_(*factor_range)
    warped = torch.empty(B, L, C, device=x.device, dtype=x.dtype)
    for b in range(B):
        new_L = max(1, min(int((factors[b] * L).item()), L))
        ib = F.interpolate(x[b].T.unsqueeze(0), size=new_L, mode='linear', align_corners=True).squeeze(0).T
        if new_L < L:
            ib = F.pad(ib, (0, 0, 0, L - new_L))
        warped[b] = ib
    return warped

def emg_augment(x, time_mask, ch_mask, probs=AUG_PROBS):
    """GPU-batched augmentation for variable-length padded EMG. x: (B, L, C)."""
    B, L, C = x.shape
    device = x.device
    y = x.clone()
    if torch.rand(()) < probs["amp_global"]:
        y = y * torch.empty(B, 1, 1, device=device).uniform_(0.8, 1.2)
    if torch.rand(()) < probs["amp_per_ch"]:
        y = y * torch.empty(B, 1, C, device=device).uniform_(0.8, 1.2)
    if torch.rand(()) < probs["baseline"]:
        tlin = torch.linspace(0, 1, L, device=device, dtype=y.dtype).view(1, L, 1)
        if torch.rand(()) < 0.5:
            phase = 2 * math.pi * torch.rand(B, 1, 1, device=device, dtype=y.dtype)
            drift = 0.05 * torch.sin(phase + 2 * math.pi * 0.5 * tlin)
        else:
            slope = 0.05 * (2 * torch.rand(B, 1, 1, device=device, dtype=y.dtype) - 1)
            drift = slope * tlin
        y = y + drift
    if torch.rand(()) < probs["tshift"]:
        shift = int(torch.randint(-8, 9, ()).item())
        if shift != 0:
            for b in range(B):
                l = int(time_mask[b].sum().item())
                if l > 0:
                    y[b, :l] = torch.roll(y[b, :l], shifts=shift, dims=0)
    y = _time_warp(y, prob=probs["time_warp"])
    y = _magnitude_warp(y, prob=probs["mag_warp"])
    if torch.rand(()) < probs["noise"]:
        std = y.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        y = y + 0.02 * std * torch.randn_like(y)
    if torch.rand(()) < probs["ch_dropout"]:
        k = max(1, int(C * 0.1))
        for b in range(B):
            valid = torch.where(ch_mask[b])[0]
            if valid.numel() > 0:
                y[b, :, valid[torch.randperm(valid.numel(), device=device)[:k]]] = 0
    if torch.rand(()) < probs["ch_perm"]:
        for b in range(B):
            valid = torch.where(ch_mask[b])[0]
            if valid.numel() > 1:
                perm = valid[torch.randperm(valid.numel(), device=device)]
                y_b = y[b].clone()
                y[b, :, valid] = y_b[:, perm]
    if torch.rand(()) < probs["lowpass"]:
        k = int(torch.randint(0, 3, (1,), device=device).item() * 2 + 3)
        y = _lowpass_avg(y, k=k)
    return y

def augment_gpu(x):
    """Simple batch augmentation for fixed-shape windows. x: (B, L, C)."""
    y = x.clone()
    device = x.device
    if torch.rand((), device=device) < 0.5:
        y = y * (1.0 + 0.1 * torch.randn((), device=device, dtype=x.dtype))
    if torch.rand((), device=device) < 0.5:
        y = torch.roll(y, shifts=int(torch.randint(-4, 5, (), device=device)), dims=1)
    if torch.rand((), device=device) < 0.5:
        std = y.std().clamp_min(1e-6)
        y = y + 0.02 * std * torch.randn_like(y)
    if torch.rand((), device=device) < 0.1:
        k = max(1, int(torch.randint(1, min(2, y.shape[2]), (), device=device)))
        y[:, :, torch.randperm(y.shape[2], device=device)[:k]] = 0
    return y

# ======== DATASETS ========

_ws_pat = re.compile(r"_ws(\d+)_?(\d+)?_")

def _parse_ws(fname):
    m = _ws_pat.search(fname)
    if not m:
        return None
    a, b = m.group(1), m.group(2)
    return float(a) if b is None else float(f"{a}.{b}")

def _infer_fs(n_samples, ws):
    if ws is None or ws <= 0:
        return None
    return float(n_samples) / float(ws)

class NPYWindows(Dataset):
    """Memory-mapped dataset over chunked .npy window files from process_cross_dataset.py.

    Each file has shape (N, L, C) in float16. Files from different datasets and window
    sizes are interleaved; channels and lengths may vary across files.
    """

    def __init__(self, root, subset_ratio=1.0, ws_filter=None):
        files = sorted(glob.glob(os.path.join(root, "*.npy")))
        if not files:
            raise FileNotFoundError(f"No .npy files in {root}")
        if ws_filter is not None:
            files = [f for f in files if abs(_parse_ws(os.path.basename(f)) - ws_filter) < 1e-4
                     if _parse_ws(os.path.basename(f)) is not None]
        self.files, self.maps, self.LC, self.ws_list, self.fs_list, self.lengths = [], [], [], [], [], []
        all_Cs = []
        for f in tqdm(files, desc="Scanning files", leave=False):
            ws = _parse_ws(os.path.basename(f))
            try:
                arr = np.load(f, mmap_mode="r")
            except Exception as e:
                print(f"Warning: skipping {f}: {e}")
                continue
            if arr.ndim != 3 or arr.shape[0] == 0:
                continue
            N, L, C = arr.shape
            all_Cs.append(C)
            keep = int(N * subset_ratio)
            if keep <= 0:
                continue
            self.files.append(f)
            self.maps.append(arr)
            self.LC.append((L, C))
            self.ws_list.append(ws)
            self.fs_list.append(_infer_fs(L, ws))
            self.lengths.append(keep)
        if not self.files:
            raise RuntimeError("No usable files found.")
        self.cum = np.cumsum([0] + self.lengths)
        self.global_max_C = int(max(all_Cs))

    def __len__(self):
        return int(self.cum[-1])

    def _locate(self, idx):
        i = bisect.bisect_right(self.cum, idx) - 1
        return i, idx - self.cum[i]

    def __getitem__(self, idx):
        fi, row = self._locate(idx)
        x = torch.from_numpy(self.maps[fi][row].astype(np.float16, copy=False))
        fs = self.fs_list[fi] if self.fs_list[fi] is not None else 1.0
        return {"x": x, "fs": float(fs)}


def collate_variable(batch, global_max_C):
    """Collate variable-length, variable-channel samples into padded tensors."""
    max_L = max(item["x"].shape[0] for item in batch)
    B = len(batch)
    xs = torch.zeros(B, max_L, global_max_C, dtype=torch.float16)
    time_masks = torch.zeros(B, max_L, dtype=torch.bool)
    ch_masks = torch.zeros(B, global_max_C, dtype=torch.bool)
    fs_list = torch.zeros(B, dtype=torch.float32)
    for b, item in enumerate(batch):
        x = item["x"]
        L, C = x.shape
        xs[b, :L, :C] = x
        time_masks[b, :L] = True
        ch_masks[b, :C] = True
        fs_list[b] = item["fs"]
    return xs, time_masks, ch_masks, fs_list


class FixedWindowNPY(Dataset):
    """Fixed-shape dataset loading files for a single window size."""

    def __init__(self, root, ws_filter, split_frac=0.9, split="train", seed=SEED):
        files = sorted(glob.glob(os.path.join(root, "*.npy")))
        files = [
            f for f in files
            if (ws := _parse_ws(os.path.basename(f))) is not None
            and abs(ws - ws_filter) < 1e-4
        ]
        if not files:
            raise FileNotFoundError(f"No files matching ws={ws_filter} in {root}")
        self.maps = [np.load(f, mmap_mode="r") for f in files]
        all_idx = [(fi, i) for fi, m in enumerate(self.maps) for i in range(m.shape[0])]
        rng = np.random.default_rng(seed)
        rng.shuffle(all_idx)
        cut = int(len(all_idx) * split_frac)
        self.index = all_idx[:cut] if split == "train" else all_idx[cut:]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, i = self.index[idx]
        return torch.from_numpy(self.maps[fi][i].astype(np.float32, copy=False))

# ======== LOADERS ========

def create_fsdp_loaders(data_path, batch_size, world_size, rank, subset_ratio=1.0):
    """Loaders for variable-length FSDP pretraining."""
    ds = NPYWindows(data_path, subset_ratio=subset_ratio)
    gen = torch.Generator().manual_seed(SEED)
    train_len = int(0.9 * len(ds))
    val_len = len(ds) - train_len
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=gen)
    collate_fn = functools.partial(collate_variable, global_max_C=ds.global_max_C)

    def sampler(d, shuffle):
        if world_size <= 1:
            return None
        return torch.utils.data.distributed.DistributedSampler(
            d, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=SEED
        )

    tr_s, va_s = sampler(train_ds, True), sampler(val_ds, False)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=(tr_s is None), sampler=tr_s,
        num_workers=8, prefetch_factor=2,
        pin_memory=True, drop_last=True,
        persistent_workers=True, worker_init_fn=_worker_init_fn,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, sampler=va_s,
        num_workers=4, prefetch_factor=2,
        pin_memory=True, drop_last=False,
        persistent_workers=True, worker_init_fn=_worker_init_fn,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, tr_s, ds.global_max_C


def create_ddp_loaders(data_path, batch_size, ws_filter, world_size=1, rank=0):
    """Loaders for fixed-shape DDP pretraining."""
    train_ds = FixedWindowNPY(data_path, ws_filter=ws_filter, split="train")
    val_ds = FixedWindowNPY(data_path, ws_filter=ws_filter, split="val")

    def sampler(d, shuffle):
        if world_size <= 1:
            return None
        return torch.utils.data.distributed.DistributedSampler(
            d, num_replicas=world_size, rank=rank, shuffle=shuffle
        )

    tr_s, va_s = sampler(train_ds, True), sampler(val_ds, False)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=(tr_s is None), sampler=tr_s,
        num_workers=8, prefetch_factor=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, sampler=va_s,
        num_workers=4, prefetch_factor=4, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader, tr_s, va_s

# ======== CHECKPOINTING ========

def save_checkpoint(model, path, epoch=None, optimizer=None,
                    scheduler=None, scaler=None, best_val=None, extra=None):
    from torch.nn.parallel import DistributedDataParallel as DDP
    state = {
        "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if extra is not None:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(model, path, rank=0, optimizer=None, scheduler=None,
                    scaler=None, map_location="cpu"):
    from torch.nn.parallel import DistributedDataParallel as DDP
    ckpt = torch.load(path, map_location=map_location)
    m = model.module if isinstance(model, DDP) else model
    m.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    if rank == 0:
        print(f"Loaded {path} | epoch={ckpt.get('epoch')} best_val={ckpt.get('best_val'):.4f}")
    return ckpt.get("epoch", 0), ckpt.get("best_val", float("inf"))
