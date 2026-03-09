import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, balanced_accuracy_score
from typing import Any

import torch; import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.amp import GradScaler, autocast
from torch.utils.data import (DataLoader,TensorDataset)

def is_notebook():
    try:
        from IPython import get_ipython; shell = get_ipython()
        if shell is None: return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except: return False

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from libemg.feature_extractor import FeatureExtractor
from Losses.VICReg import vicreg_loss, augment


# ======== CONFIG ========
PATH = "pickles"
DTYPE = np.float32
SEQ = 200; SSL_INC = 40; INC = 5; CH = 8; CLASSES = 5
VAL_CUTOFF = 55; WORKERS = 4; PRE_FETCH = 2; VERBOSE=True
UPDATE_EVERY = 1; PRESIST_WORKER = True; PIN_MEMORY = True
DEVICE = 'cuda'

FT_CLASSES = [0, 1, 2, 3, 4]

SSL_EPOCHS = 20; SSL_LR = 5e-5; LR_PATIENCE_SSL = 4
FT_EPOCHS = 100; LR_INIT = 1e-3; LR_MIN = 5e-6
LR_FACTOR = 0.8; LR_PATIENCE = 4; DROPOUT = 0.2 
SSL_BATCH_SIZE = 4096; BATCH_SIZE = 128; PATIENCE = 10


# ======== UTILS ========
def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def remap_labels(y: np.ndarray, keep_classes: list[int]) -> np.ndarray:
    lut = {c: i for i, c in enumerate(keep_classes)}
    return np.vectorize(lut.get)(y).astype(np.int64)

def filter_by_classes(x: np.ndarray, y: np.ndarray, 
                      keep_classes: list[int]):
    keep = np.isin(y, np.array(keep_classes, dtype=y.dtype))
    return x[keep], y[keep]

def _check(name, t):
    if not torch.is_tensor(t): return
    if torch.isnan(t).any() or torch.isinf(t).any():
        raise RuntimeError(f"NaN/Inf in {name}: "
                           f"nan={torch.isnan(t).any().item()} "
                           f"inf={torch.isinf(t).any().item()} "
                           f"min={t.nan_to_num().min().item()} "
                           f"max={t.nan_to_num().max().item()}")


# ======== UTILS ========
def extract_features(x, feature_list, feature_dic=None):
    feature_extractor = FeatureExtractor()
    features = feature_extractor.extract_features(feature_list, x, array=True,
                                fix_feature_errors=False, feature_dic=feature_dic)
    return torch.from_numpy(features.astype(DTYPE))


# ======== LOADERS ========
def create_sup_loader(x, y, batch=BATCH_SIZE, shuffle=False, 
                  workers=WORKERS, prefetch_factor=PRE_FETCH,
                  persistent_workers=PRESIST_WORKER):
    return DataLoader(
    TensorDataset(torch.from_numpy(x), 
                  torch.from_numpy(y)),
                #   torch.tensor(x), 
                #   torch.tensor(y)),
    batch_size=batch,
    shuffle=shuffle,
    num_workers=workers,
    prefetch_factor=prefetch_factor if workers > 0 else None,
    persistent_workers=persistent_workers,
    pin_memory=PIN_MEMORY,
    drop_last=False)


def create_ssl_loader(x, batch=BATCH_SIZE, shuffle=False, 
                  workers=WORKERS, prefetch_factor=PRE_FETCH,
                  persistent_workers=PRESIST_WORKER):
    return DataLoader(
    TensorDataset(torch.from_numpy(x)),
                #   torch.tensor(x)), 
    batch_size=batch,
    shuffle=shuffle,
    num_workers=workers,
    prefetch_factor=prefetch_factor if workers > 0 else None,
    persistent_workers=persistent_workers,
    pin_memory=PIN_MEMORY,
    drop_last=False)


# ======== TRAIN (VICREG SSL) ========
def pretrain_vicreg(
    model: nn.Module,
    ssl_loader: DataLoader,
    name: str,
    feature_list: list=None,
    feature_dict: dict=None,
    epochs: int = SSL_EPOCHS,
    lr: float = SSL_LR,
    min_lr: float = LR_MIN,
    lr_factor: float = LR_FACTOR,
    lr_patience: int = LR_PATIENCE_SSL,
    verbose=VERBOSE,
    device: str = DEVICE):

    model.to(device)
    opt = Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr)
    scaler = GradScaler(enabled=(device == "cuda"))

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        total = 0
        step = 0
        pbar = tqdm(total=len(ssl_loader), desc=f"{name} | SSL Ep {ep}", 
                    leave=True, dynamic_ncols=True, disable=not verbose)

        for (xb,) in ssl_loader:
            xb = xb.to(device, non_blocking=True)
            x1 = augment(xb)
            x2 = augment(xb)

            if feature_list is not None:
                x1 = extract_features(x1.detach().cpu(), feature_list, feature_dict)
                x2 = extract_features(x2.detach().cpu(), feature_list, feature_dict)
                x1 = x1.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", 
                          enabled=(device == "cuda")):
                z1 = model(x1, return_proj=True)
                z2 = model(x2, return_proj=True)
                loss = vicreg_loss(z1, z2)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total += xb.numel()
            step += 1
            total_loss += loss.detach()

            if not(step % UPDATE_EVERY):
                pbar.update(UPDATE_EVERY)
                pbar.set_postfix(loss=f"{total_loss.item() / step:10.8f}", 
                                 LR=f"{opt.param_groups[0]['lr']:8.6f}")

        if step % UPDATE_EVERY:
            pbar.update(step % UPDATE_EVERY)

        epoch_loss = total_loss.item() / max(1, len(ssl_loader))
        sch.step(epoch_loss)
        pbar.close()

    return model


# ======== TRAIN (SUP FINETUNE) ========
def train_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    name: str,
    loss_fn: Any,
    epochs: int = FT_EPOCHS,
    lr: float = LR_INIT,
    min_lr: float = LR_MIN,
    lr_factor: float = LR_FACTOR,
    lr_patience: int = LR_PATIENCE,
    patience: int = PATIENCE,
    verbose=VERBOSE,
    device: str = DEVICE):

    model.to(device)
    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr)
    scaler = GradScaler(enabled=(device == "cuda"))

    best_val = 1e9
    best_state = {k: v.clone().cpu() for k, v in 
                  model.state_dict().items()}
    wait = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        correct = torch.tensor(0.0, device=device)
        total = 0
        step = 0
        pbar = tqdm(total=len(train_loader), desc=f"{name} | FT Ep {ep}", 
                    leave=True, dynamic_ncols=True, disable=not verbose)

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model(xb)
                loss = loss_fn(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += loss.detach()
            correct += (logits.argmax(1) == yb).sum()
            total += yb.numel()
            step += 1

            if not(step % UPDATE_EVERY):
                pbar.update(UPDATE_EVERY)
                pbar.set_postfix(
                    loss=f"{total_loss.item() / step:10.8f}",
                    acc=f"{correct.item() / max(1, total):6.4f}",
                    LR=f"{opt.param_groups[0]['lr']:8.6f}")

        if step % UPDATE_EVERY:
            pbar.update(step % UPDATE_EVERY)

        val_acc, val_loss, _, _ = evaluate_sup(model, val_loader, loss_fn, device)
        sch.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone().cpu() for k, v in 
                          model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                tqdm.write(f"{name} | Early stop")
                pbar.close()
                break

        pbar.set_postfix(
            loss=f"{total_loss.item() / max(1, len(train_loader)):10.6f}",
            acc=f"{correct.item() / max(1, total):6.4f}",
            val_loss=f"{val_loss:10.6f}",
            val_acc=f"{val_acc:6.4f}",
            LR=f"{opt.param_groups[0]['lr']:8.6f}",
            wait=f"{wait:3.0f}")
        pbar.close()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ======== EVAL (SUPERVISED) ========
@torch.no_grad()
def evaluate_sup(model, loader, loss_fn, device):
    model.eval()
    model.to(device)
    lsum = torch.tensor(0.0, device=device)
    cor = torch.tensor(0.0, device=device)
    tot = 0
    y_true_list, y_pred_list = [], []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", 
                                enabled=(device == "cuda")):
            logits = model(xb)
            loss = loss_fn(logits, yb)
        preds = logits.argmax(1)
        lsum += loss
        cor += (preds == yb).sum()
        tot += yb.numel()
        y_true_list.append(yb)
        y_pred_list.append(preds)

    y_true = torch.cat(y_true_list).cpu().numpy()
    y_pred = torch.cat(y_pred_list).cpu().numpy()
    f1 = f1_score(y_true, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    avg_acc = cor.item() / max(1, tot)
    avg_loss = lsum.item() / max(1, len(loader))
    return avg_acc, avg_loss, f1, bal_acc