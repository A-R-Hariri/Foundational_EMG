import os, re, math, warnings, glob, bisect, functools
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
import torch.distributed as dist
from tqdm import tqdm

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing
)

# ==================== CONFIG ====================
DATA_PATH     = "data_pickles"
CKPT_PATH     = "pretrain_ckpts"
os.makedirs(CKPT_PATH, exist_ok=True)
BATCH_SIZE    = 128
EPOCHS        = 200
LR            = 1e-4
MIN_LR        = 1e-6
RL_FACTOR     = 0.8
RL_PATIENCE   = 2
PATIENCE      = 10
LATENT_DIM    = 256
NUM_HEADS     = 4
NUM_LAYERS    = 4
DROPOUT       = 0.1
CONV_KERNEL = 8
CONV_STRIDE = 4
CONV_PADDING = CONV_KERNEL // 2
AUG_PROBS = {
    "amp_global": 0.7, "amp_per_ch": 0.5, "baseline":   0.3, "tshift":     0.7,
    "time_warp":  0.5, "noise":      0.6, "ch_dropout": 0.2, "ch_perm":    0.5,
    "mag_warp":   0.5, "lowpass":    0.3,
}
MAE_MASK_FRAC = 0.3
MAE_LOSS_W    = 1.0
VIC_LOSS_W    = 1.0
SEED          = 67
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
# ===============================================

_ws_pat = re.compile(r"_ws(\d+)_?(\d+)?_")
def parse_ws_from_name(fname: str):
    m = _ws_pat.search(fname)
    if not m: return None
    a, b = m.group(1), m.group(2)
    return float(a) if b is None else float(f"{a}.{b}")

def infer_fs_from_shape_and_ws(L: int, ws: float):
    if ws is None or ws <= 0: return None
    return float(L) / (float(ws) + 1e-9)

class NPYWindows(Dataset):
    def __init__(self, root: str, subset_ratio: float = 1.0):
        files = sorted(glob.glob(os.path.join(root, "*.npy")))
        if not files:
            raise FileNotFoundError(f"No .npy files in {root}. Did data2.py run?")
        self.files, self.maps, self.LC, self.ws, self.fs, self.lengths = [], [], [], [], [], []
        
        all_Cs = []
        for f in tqdm(files, desc="Scanning dataset files..."):
            ws = parse_ws_from_name(os.path.basename(f))
            try:
                arr = np.load(f, mmap_mode="r")
            except Exception as e:
                print(f"\nWarning: Could not load {f}, skipping. Error: {e}")
                continue
            if arr.ndim != 3 or arr.shape[0] == 0:
                print(f"\nWarning: Skipping {f}, invalid shape {arr.shape}")
                continue
            N, L, C = arr.shape
            all_Cs.append(C)
            keep = int(N * subset_ratio)
            if keep <= 0: continue
            self.files.append(f)
            self.maps.append(arr)
            self.LC.append((L, C))
            self.ws.append(ws)
            self.fs.append(infer_fs_from_shape_and_ws(L, ws))
            self.lengths.append(keep)
        if not self.files:
            raise RuntimeError("No usable files after subset filtering.")
        self.cum = np.cumsum([0] + self.lengths)
        self.global_max_C = int(max(all_Cs)) if all_Cs else 0
        if self.global_max_C == 0:
             raise RuntimeError("Could not determine global_max_C.")

    def __len__(self):
        return int(self.cum[-1])

    def _locate(self, idx: int):
        i = bisect.bisect_right(self.cum, idx) - 1
        row = idx - self.cum[i]
        return i, row

    def __getitem__(self, idx: int):
        fi, row = self._locate(idx)
        arr = self.maps[fi][row]
        x = torch.from_numpy(arr.astype(np.float16, copy=False))
        fs = self.fs[fi] if self.fs[fi] is not None else 1.0
        return {"x": x, "fs": float(fs)}

def _worker_init_fn(worker_id):
    base_seed = SEED
    np.random.seed(base_seed + worker_id + 1)
    torch.manual_seed(base_seed + worker_id + 1)

def collate_variable(batch, global_max_C: int):
    max_L = max(item["x"].shape[0] for item in batch)
    max_C = global_max_C
    B = len(batch)
    xs = torch.zeros(B, max_L, max_C, dtype=torch.float16)
    time_masks = torch.zeros(B, max_L, dtype=torch.bool)
    ch_masks   = torch.zeros(B, max_C, dtype=torch.bool)
    fs_list    = torch.zeros(B, dtype=torch.float32)
    for b, item in enumerate(batch):
        x = item["x"]
        L, C = x.shape
        xs[b, :L, :C] = x
        time_masks[b, :L] = True
        ch_masks[b, :C]   = True
        fs_list[b]        = item["fs"]
    return xs, time_masks, ch_masks, fs_list

def lowpass_avg(x, k=5):
    if k <= 1: return x
    csum = torch.cumsum(x, dim=1)
    ma = csum.clone()
    ma[:, k:, :] = csum[:, k:, :] - csum[:, :-k, :]
    ma = ma / k
    if k > 1:
        head_counts = torch.arange(1, k, device=x.device, dtype=x.dtype).view(1, -1, 1)
        ma[:, :k-1, :] = csum[:, :k-1, :] / head_counts
    return ma

def magnitude_warp(x, prob=0.5, knots=4, amp=0.2):
    B, L, C = x.shape
    if torch.rand(()) >= prob or knots <= 1: return x
    device = x.device
    scales_k = 1.0 + amp * (2 * torch.rand(B, knots, device=device) - 1.0)
    scales_k = scales_k.unsqueeze(1)
    scales_l = F.interpolate(scales_k, size=L, mode='linear', align_corners=True)
    warp = scales_l.squeeze(1).unsqueeze(-1)
    return x * warp

def time_warp(x, prob=0.5, factor_range=(0.95, 1.05)):
    B, L, C = x.shape
    if torch.rand(()) >= prob: return x
    device = x.device
    factors = torch.empty(B, device=device).uniform_(*factor_range)
    warped = torch.empty(B, L, C, device=device, dtype=x.dtype)
    for b in range(B):
        new_L = int((factors[b] * L).item())
        new_L = max(1, min(new_L, L))
        xb = x[b].permute(1,0).unsqueeze(0)
        ib = F.interpolate(xb, size=new_L, mode='linear', align_corners=True).squeeze(0).permute(1,0)
        if new_L < L: ib = F.pad(ib, (0,0,0, L-new_L))
        else:         ib = ib[:L]
        warped[b] = ib
    return warped

def emg_augment(x, time_mask, ch_mask, probs=AUG_PROBS):
    B, L, C = x.shape
    device = x.device
    y = x.clone()
    if torch.rand(()) < probs["amp_global"]:
        scale = torch.empty(B, 1, 1, device=device).uniform_(0.8, 1.2)
        y = y * scale
    if torch.rand(()) < probs["amp_per_ch"]:
        scale_ch = torch.empty(B, 1, C, device=device).uniform_(0.8, 1.2)
        y = y * scale_ch
    if torch.rand(()) < probs["baseline"]:
        if torch.rand(()) < 0.5:
            phase = 2*math.pi*torch.rand(B, 1, 1, device=device, dtype=y.dtype)
            tlin = torch.linspace(0, 1, L, device=device, dtype=y.dtype).view(1, L, 1)
            drift = 0.05 * torch.sin(phase + 2*math.pi*0.5*tlin)
        else:
            slope = 0.05 * (2 * torch.rand(B, 1, 1, device=device, dtype=y.dtype) - 1)
            tlin = torch.linspace(0, 1, L, device=device, dtype=y.dtype).view(1, L, 1)
            drift = slope * tlin
        y = y + drift
    if torch.rand(()) < probs["tshift"]:
        shift = int(torch.randint(-8, 9, ()).item())
        if shift != 0:
            for b in range(B):
                l = int(time_mask[b].sum().item())
                if l > 0:
                    y[b, :l] = torch.roll(y[b, :l], shifts=shift, dims=0)
    y = time_warp(y, prob=probs["time_warp"])
    y = magnitude_warp(y, prob=probs["mag_warp"])
    if torch.rand(()) < probs["noise"]:
        std = y.std(dim=(1,2), keepdim=True).clamp_min(1e-6)
        y = y + 0.02 * std * torch.randn_like(y)
    if torch.rand(()) < probs["ch_dropout"]:
        k = max(1, int(C * 0.1))
        for b in range(B):
            valid = torch.where(ch_mask[b])[0]
            if valid.numel() > 0:
                perm = torch.randperm(valid.numel(), device=device)
                drop = valid[perm[:k]]
                y[b, :, drop] = 0
    if torch.rand(()) < probs["ch_perm"]:
        for b in range(B):
            valid = torch.where(ch_mask[b])[0]
            if valid.numel() > 1:
                perm = valid[torch.randperm(valid.numel(), device=device)]
                y_b = y[b].clone()
                y[b, :, valid] = y_b[:, perm]
    if torch.rand(()) < probs["lowpass"]:
        k = int(torch.randint(0, 3, (1,), device=device).item() * 2 + 3)
        y = lowpass_avg(y, k=k)
    return y


class SinusoidalTimeEncoding(nn.Module):
    def __init__(self, d_pe=64):
        super().__init__()
        self.d_pe = d_pe
    def forward(self, times_sec):
        B, L = times_sec.shape
        t = times_sec.unsqueeze(-1)
        i = torch.arange(self.d_pe // 2, device=times_sec.device, dtype=torch.float32)
        freqs = 1.0 / (10000 ** (2 * i / self.d_pe))
        pe = t * freqs
        pe = torch.cat([torch.sin(pe), torch.cos(pe)], dim=-1)
        return pe

class Encoder(nn.Module):
    def __init__(self, max_C, d_model=LATENT_DIM, kernel_size=CONV_KERNEL, stride=CONV_STRIDE, padding=CONV_PADDING, d_pe=64):
        super().__init__()
        self.d_model = d_model
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.phi = nn.Conv1d(
            in_channels=max_C, out_channels=d_model, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=True,
        )
        self.pos_encoder = SinusoidalTimeEncoding(d_pe=d_pe)
        self.pe_proj = nn.Linear(d_model + d_pe, d_model)
        self.mask_token = nn.Parameter(torch.randn(d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, NUM_HEADS, dim_feedforward=4*d_model,
            batch_first=True, norm_first=True, dropout=DROPOUT
        )
        self.transformer = nn.TransformerEncoder(layer, NUM_LAYERS)

    def forward(self, x, time_mask, ch_mask, times_sec, mae_mask=None):
        B, L, C = x.shape
        if ch_mask is not None:
            x = x * ch_mask.unsqueeze(1)
        x = x.transpose(1, 2)
        with autocast(device_type="cuda", enabled=False):
            x = self.phi(x.float())
        x = x.transpose(1, 2)
        L_new = x.size(1)
        with torch.no_grad():
            time_mask_new = F.max_pool1d(
                 time_mask.float().unsqueeze(1), kernel_size=self.kernel_size,
                 stride=self.stride, padding=self.padding
            ).squeeze(1).bool()
            if time_mask_new.size(1) > L_new:
                time_mask_new = time_mask_new[:, :L_new]
            elif time_mask_new.size(1) < L_new:
                pad = L_new - time_mask_new.size(1)
                tmp = torch.zeros(time_mask_new.size(0), L_new, device=time_mask_new.device, dtype=torch.bool)
                tmp[:, :time_mask_new.size(1)] = time_mask_new
                time_mask_new = tmp
        pe = self.pos_encoder(times_sec)
        pe = F.interpolate(pe.permute(0,2,1), size=L_new, mode='linear', align_corners=False).permute(0,2,1)
        x = torch.cat([x, pe], dim=-1)
        x = self.pe_proj(x)
        if mae_mask is not None:
            with torch.no_grad():
                mae_mask_new = F.max_pool1d(
                    mae_mask.float().unsqueeze(1), kernel_size=self.kernel_size,
                    stride=self.stride, padding=self.padding
                ).squeeze(1).bool()
                if mae_mask_new.size(1) > L_new:
                    mae_mask_new = mae_mask_new[:, :L_new]
                elif mae_mask_new.size(1) < L_new:
                    pad = L_new - mae_mask_new.size(1)
                    tmp = torch.zeros(mae_mask_new.size(0), L_new, device=mae_mask_new.device, dtype=torch.bool)
                    tmp[:, :mae_mask_new.size(1)] = mae_mask_new
                    mae_mask_new = tmp
            mask_tok = self.mask_token.view(1,1,-1).expand(B, L_new, -1).to(x.dtype)
            x = torch.where(mae_mask_new.unsqueeze(-1), mask_tok, x)
        z_seq = self.transformer(x, src_key_padding_mask=~time_mask_new)
        return z_seq, time_mask_new

class VICHead(nn.Module):
    def __init__(self, d_model=LATENT_DIM, proj=LATENT_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, proj), nn.ReLU(), nn.Linear(proj, proj)
        )
    def forward(self, z_seq, time_mask_new):
        denom = time_mask_new.sum(dim=1).clamp_min(1)
        pooled = (z_seq * time_mask_new.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(-1)
        return self.mlp(pooled)

class MAEDecoder(nn.Module):
    def __init__(self, max_C, d_model=LATENT_DIM, stride=CONV_STRIDE):
        super().__init__()
        self.stride = stride
        self.decoder_proj = nn.Linear(d_model, max_C)
    def forward(self, z_seq, L_orig, C_max):
        y = self.decoder_proj(z_seq)
        y = y.transpose(1, 2)
        y = F.interpolate(y, size=L_orig, mode='linear', align_corners=False)
        y = y.transpose(1, 2)
        return y

def vicreg_loss(z1, z2, lamb=25.0, mu=25.0, nu=1.0, gamma=1.0):
    sim = ((z1 - z2) ** 2).mean()
    def var_term(z):
        std = z.std(dim=0, unbiased=False)
        return F.relu(gamma - std).mean()
    v = var_term(z1) + var_term(z2)
    def cov_term(z):
        zc = z - z.mean(dim=0)
        cov = (zc.T @ zc) / (zc.size(0) - 1 + 1e-6)
        off = cov - torch.diag(torch.diag(cov))
        return (off ** 2).mean()
    c = cov_term(z1) + cov_term(z2)
    return lamb * sim + mu * v + nu * c

def generate_time_mask(time_mask, frac=MAE_MASK_FRAC):
    B, L = time_mask.shape
    mask = torch.zeros_like(time_mask)
    for b in range(B):
        valid_idx = torch.where(time_mask[b])[0]
        k = int(max(1, frac * valid_idx.numel()))
        if k > 0:
            sel = valid_idx[torch.randperm(valid_idx.numel(), device=time_mask.device)[:k]]
            mask[b, sel] = True
    return mask

class EMGPretrainer(nn.Module):
    def __init__(self, max_C, device=torch.device("cpu")):
        super().__init__()
        self.enc = Encoder(max_C=max_C, d_model=LATENT_DIM)
        self.vic = VICHead()
        self.mae = MAEDecoder(max_C=max_C)
    def forward_encoder(self, x, time_mask, ch_mask, times_sec, mae_mask=None):
        return self.enc(x, time_mask, ch_mask, times_sec, mae_mask=mae_mask)
    def project_vic(self, z_seq, time_mask_new):
        return self.vic(z_seq, time_mask_new)
    def decode_mae(self, z_seq, L_orig, C_max):
        return self.mae(z_seq, L_orig, C_max)


def sync_mean(val, device):
    t = torch.tensor([float(val)], device=device, dtype=torch.float32)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t / dist.get_world_size()
    return float(t.item())

def run_epoch(epoch, model, loader, optimizer, scaler, device, train=True):
    model.train(train)
    running_loss, running_vic, running_mae, valid_iters = 0.0, 0.0, 0.0, 0
    etype = 'Train' if train else 'Val'
    show_bar = (not dist.is_initialized()) or dist.get_rank() == 0
    
    rank = dist.get_rank() if dist.is_initialized() else 0

    pbar = tqdm(total=len(loader), desc=f"{etype} - Ep: {epoch}", leave=False, dynamic_ncols=True,
                bar_format='{l_bar}{bar:20}{r_bar}',
                mininterval=1, disable=not show_bar)

    for step_idx, (xs, time_masks, ch_masks, fs_list) in enumerate(loader):
        xs = xs.to(device, non_blocking=True).to(torch.float32)
        time_masks = time_masks.to(device, non_blocking=True)
        ch_masks   = ch_masks.to(device, non_blocking=True)
        fs_list    = fs_list.to(device, non_blocking=True)
        
        xs = torch.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            B, L, C = xs.shape
            t_idx = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0).expand(B, L)
            times_sec = t_idx / fs_list.unsqueeze(1).clamp_min(1e-6)
            
            x1 = emg_augment(xs, time_masks, ch_masks)
            x2 = emg_augment(xs, time_masks, ch_masks)
            
            with torch.no_grad():
                x1_norm = x1 / x1.std(dim=(1,2), keepdim=True).clamp_min(1e-5) 
            
            mae_mask = generate_time_mask(time_masks, frac=MAE_MASK_FRAC).to(device)

            with autocast(device_type="cuda", enabled=True):
                z1_seq, m1 = model.forward_encoder(x1, time_masks, ch_masks, times_sec, mae_mask=None)
                z2_seq, m2 = model.forward_encoder(x2, time_masks, ch_masks, times_sec, mae_mask=None)
                z1_proj = model.project_vic(z1_seq, m1)
                z2_proj = model.project_vic(z2_seq, m2)
                loss_vic = vicreg_loss(z1_proj, z2_proj)
                z_mask_seq, _ = model.forward_encoder(x1, time_masks, ch_masks, times_sec, mae_mask=mae_mask)
                recon = model.decode_mae(z_mask_seq, L, C)
                valid_mask = time_masks.unsqueeze(-1) & ch_masks.unsqueeze(1)
                mae_pos = mae_mask.unsqueeze(-1) & valid_mask
                if mae_pos.any():
                    recon_norm = recon / recon.std(dim=(1,2), keepdim=True).clamp_min(1e-5)
                    diff = (recon_norm - x1_norm) 
                    loss_mae = (diff[mae_pos] ** 2).mean()
                else:
                    loss_mae = torch.zeros((), device=device, dtype=torch.float32)
                loss = VIC_LOSS_W * loss_vic + MAE_LOSS_W * loss_mae
            
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss is {loss.item()}")

            current_loss_val = loss.detach().item()
            current_vic_val = loss_vic.detach().item()
            current_mae_val = loss_mae.detach().item()

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

        except Exception as e:
            current_loss_val = float('nan')
            current_vic_val = float('nan')
            current_mae_val = float('nan')
            if rank == 0:
                pbar.write(f"\nWarning: Rank {rank} failed step {step_idx}. Error: {e}. Skipping step.")
            if train:
                optimizer.zero_grad(set_to_none=True)

        if math.isfinite(current_loss_val):
            running_loss += current_loss_val
            running_vic += current_vic_val
            running_mae += current_mae_val
            valid_iters += 1
            
        # ======== Debug ========   
        if step_idx == 10: 
            if rank == 0:
                pbar.write("\nDEBUG: Reached 11 steps, breaking epoch.")
            break 

        pbar.update(1)
        if show_bar:
            pbar.set_postfix({
                "loss": f"{running_loss / max(1, valid_iters):.4f}",
                "vic": f"{running_vic / max(1, valid_iters):.4f}",
                "mae": f"{running_mae / max(1, valid_iters):.4f}"})
            
    pbar.close()
    
    if valid_iters == 0:
        return 0.0 
    return running_loss / valid_iters


def main():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )
        rank   = dist.get_rank()
        world  = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        distributed = True
    else:
        rank = 0
        world = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        distributed = False

    ds = NPYWindows(DATA_PATH, subset_ratio=1.0)
    if rank == 0:
        print(f"Dataset loaded: {len(ds)} samples. Global Max Channels: {ds.global_max_C}")

    gen = torch.Generator().manual_seed(SEED)
    train_len = int(0.9 * len(ds))
    val_len   = len(ds) - train_len
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=gen)

    collate_fn = functools.partial(collate_variable, global_max_C=ds.global_max_C)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=SEED) if distributed else None
    val_sampler   = torch.utils.data.distributed.DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False) if distributed else None

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, 
                              shuffle=False if distributed else True,
                              num_workers=8, prefetch_factor=2, pin_memory=True,
                              collate_fn=collate_fn,
                              drop_last=True,
                              persistent_workers=True, worker_init_fn=_worker_init_fn,
                              sampler=train_sampler)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, prefetch_factor=2, pin_memory=True,
                            collate_fn=collate_fn,
                            drop_last=False,
                            persistent_workers=True, worker_init_fn=_worker_init_fn,
                            sampler=val_sampler)

    model = EMGPretrainer(max_C=ds.global_max_C)

    if distributed:
        modules_to_wrap = {
            Encoder, VICHead, MAEDecoder, nn.TransformerEncoderLayer,
        }
        auto_wrap_policy = ModuleWrapPolicy(module_classes=modules_to_wrap)
        checkpointing_policy = { nn.TransformerEncoderLayer }
        if not getattr(model, '_applied_checkpointing', False):
            if rank == 0:
                print(f"Rank {rank}: Applying activation checkpointing...")
            apply_activation_checkpointing(
                model, checkpoint_wrapper_fn=checkpoint_wrapper,
                check_fn=lambda m: isinstance(m, tuple(checkpointing_policy))
            )
            model._applied_checkpointing = True
        mp_policy = MixedPrecision(
            param_dtype=torch.float32, reduce_dtype=torch.bfloat16, 
            buffer_dtype=torch.float32,
        )
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD, 
            cpu_offload=CPUOffload(offload_params=False), 
            mixed_precision=mp_policy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=torch.cuda.current_device(),
            use_orig_params=True, 
            limit_all_gathers=True,
        )
        dist.barrier() 
    else:
        model = model.to(device)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(model, f"\nTotal parameters: {total_params:,}")

    opt = Adam(model.parameters(), lr=LR)
    sch = ReduceLROnPlateau(opt, mode="min", factor=RL_FACTOR, patience=RL_PATIENCE, min_lr=MIN_LR)
    scaler = GradScaler() 

    best = float("inf")
    ckpt_best = os.path.join(CKPT_PATH, "pretrain_best.pt")
    
    # This checkpoint loading logic is correct.
    if os.path.exists(ckpt_best):
        if distributed:
            if rank == 0:
                print(f"Loading checkpoint from {ckpt_best}...")
            full_state = torch.load(ckpt_best, map_location="cpu")

            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=False)):
                model.load_state_dict(full_state["model"])
            
            opt.load_state_dict(full_state["optimizer"])
            sch.load_state_dict(full_state["scheduler"])
            scaler.load_state_dict(full_state["scaler"])
            best = full_state.get("best", best)

            if rank == 0:
                print(f"Loaded {ckpt_best} (best={best:.4f})")
            dist.barrier()
        else:
            state = torch.load(ckpt_best, map_location=device)
            model.load_state_dict(state["model"])
            if "optimizer" in state:  opt.load_state_dict(state["optimizer"])
            if "scheduler" in state:  sch.load_state_dict(state["scheduler"])
            if "scaler" in state:     scaler.load_state_dict(state["scaler"])
            best = state.get("best", best)
            print(f"Loaded {ckpt_best} (best={best:.4f})")
            
    wait = 0
    for epoch in range(1, EPOCHS+1):
        if distributed:
            dist.barrier()
            train_sampler.set_epoch(epoch) 
        if rank == 0:
            print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss = run_epoch(epoch, model, train_loader, opt, scaler, device, train=True)
        
        with torch.no_grad(): 
            val_loss = run_epoch(epoch, model, val_loader, None, None, device, train=False)

        if distributed:
            train_loss = sync_mean(train_loss, device)
            val_loss   = sync_mean(val_loss, device)

        if rank == 0:
            print(f"train={train_loss:.4f}  val={val_loss:.4f}  lr={opt.param_groups[0]['lr']:.6f}")
        sch.step(val_loss)

        improved = val_loss < best

        if distributed:
            dist.barrier()

        if rank == 0:
            print("Saving checkpoint...")
            
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state = model.state_dict()
            
            if rank == 0:
                save_dict = {
                    "model": cpu_state, 
                    "optimizer": opt.state_dict(),
                    "scheduler": sch.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch, 
                    "best": best,
                    "global_max_C": ds.global_max_C,
                }
                torch.save(save_dict, os.path.join(CKPT_PATH, f"pretrain_latest.pt"))
                if improved:
                    best = val_loss
                    save_dict["best"] = best
                    torch.save(save_dict, ckpt_best)
                    print(f"New best model saved to {ckpt_best}")

        wait = 0 if improved else wait + 1
        if wait >= PATIENCE and rank == 0:
            print("Early stopping.")
            break
        
        # Debug break for epoch
        if epoch == 1:
             if rank == 0:
                 print("DEBUG: Breaking after 1 epoch.")
             break

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()