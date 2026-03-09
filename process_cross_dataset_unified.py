import warnings
import sys
import os
import gc
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")

import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
import math, random, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

os.environ["KERAS_BACKEND"] = "torch"
import keras
print(keras.backend.backend())

# import tensorflow as tf

from libemg.datasets import get_dataset_list
from libemg.feature_extractor import FeatureExtractor
from libemg.offline_metrics import OfflineMetrics
import libemg

import numpy as np 
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.utils.class_weight import compute_class_weight
import joblib
from sklearn.decomposition import PCA
import copy
import math
from scipy.signal import resample


PATH = 'pickles/'

TARGET_FS   = 500
TARGET_CH   = 8
WIN_RANGE   = [0.1, 0.15, 0.2, 0.3, 0.4]
STRIDE_FRAC = 0.25
N = 100_000_000
DTYPE = np.float16

MAX_WIN_SEC = WIN_RANGE[-1]
MAX_WIN_SAMPLES = int(MAX_WIN_SEC * TARGET_FS)  # fixed maximum window length

dataset_keys = list(get_dataset_list('ALL', True).keys())

for num, name in enumerate(dataset_keys[23:]):
    print(f"Processing dataset: {name}, {num+1}/{len(dataset_keys)}")

    # if name != 'EMG2POSE':
    #     continue

    # preallocate fixed-shape arrays (we’ll fill sequentially)
    X_train = np.zeros((N, MAX_WIN_SAMPLES, TARGET_CH), dtype=DTYPE)
    X_val   = np.zeros((N // 10, MAX_WIN_SAMPLES, TARGET_CH), dtype=DTYPE)
    train_idx = 0
    val_idx = 0

    try:
        ds_cls = get_dataset_list('ALL', True)[name]
        ds = ds_cls()
        fs = getattr(ds, 'sampling', 202)
        print(fs)
        data = ds.prepare_data(split=True)['Train'].isolate_data("subjects", list(range(306-30, 306)), fast=True).data \
            if name == 'EMGEPN612' else ds.prepare_data(split=False).data
        del ds

        for d in range(len(data)):
            if not d % 100:
                print(name, d / len(data), train_idx)
            sig = data[d].astype(DTYPE)

            # EMA normalization per channel
            alpha = 0.001
            mu = np.zeros(sig.shape[1], dtype=DTYPE)
            var = np.ones_like(mu)
            out = np.empty_like(sig, dtype=DTYPE)
            for i in range(sig.shape[0]):
                x = sig[i]
                mu = (1 - alpha) * mu + alpha * x
                var = (1 - alpha) * var + alpha * (x - mu) ** 2
                out[i] = (x - mu) / (np.sqrt(var) + 1e-8)
            sig = np.clip(out, -3.0, 3.0).astype(DTYPE)

            # resample to TARGET_FS
            if fs != TARGET_FS:
                scale = TARGET_FS / fs
                new_len = int(sig.shape[0] * scale)
                sig = resample(sig, new_len, axis=0)

            # unify channel count
            T_raw, C = sig.shape
            if C > TARGET_CH:
                x = torch.from_numpy(sig).contiguous().float().view(1, T_raw, C)
                x = F.interpolate(x, size=TARGET_CH, mode='linear', align_corners=False)
                sig = x.squeeze(0).cpu().numpy()
            elif C < TARGET_CH:
                pad = np.zeros((T_raw, TARGET_CH - C), dtype=sig.dtype)
                sig = np.concatenate([sig, pad], axis=1)

            # fixed windowing with padding to MAX_WIN_SAMPLES
            T = sig.shape[0]
            for ws in WIN_RANGE:
                ws_samples = int(round(ws * TARGET_FS))
                if ws_samples < 1:
                    continue
                stride_samples = max(1, int(round(STRIDE_FRAC * ws_samples)))
                for start in range(0, max(0, T - ws_samples + 1), stride_samples):
                    win = sig[start:start + ws_samples]
                    if ws_samples < MAX_WIN_SAMPLES:
                        pad = np.zeros((MAX_WIN_SAMPLES - ws_samples, TARGET_CH), dtype=win.dtype)
                        win = np.concatenate([win, pad], axis=0)
                    if random.random() < 0.9:
                        if train_idx < N:
                            X_train[train_idx] = win
                            train_idx += 1
                        else:
                            print("N reached:", name, d / len(data))
                            exit()
                    else:
                        if val_idx < N // 10:
                            X_val[val_idx] = win
                            val_idx += 1
                        else:
                            print("N reached:", name, d / len(data))
                            exit()
            del sig
            data[d] = None

        # trim unused preallocated space
        X_train = X_train[:train_idx]
        X_val = X_val[:val_idx]

        # save this dataset
        np.save(f"{PATH}/X_train_{name}.npy", X_train)
        np.save(f"{PATH}/X_val_{name}.npy", X_val)

        del X_train, X_val
        gc.collect()

    except Exception as e:
        print(f"Skipped {name}: {e}")

print("Prepared train and val windows.")




import os, joblib, numpy as np, gc

# -------- CONFIG --------
PATH = "pickles"
DTYPE = np.float16        # for 2× smaller disk and RAM
DELETE_ORIGINAL = True    # set False if you want to keep pkl copies

# -------- DISCOVER FILES --------
pkl_files = sorted(f for f in os.listdir(PATH) if f.endswith(".npy"))
print(f"Found {len(pkl_files)} pickle files.")

for n, fname in enumerate(pkl_files, 1):
    src = os.path.join(PATH, fname)
    dst = src.replace(".pkl", ".npy")

    try:
        print(f"[{n}/{len(pkl_files)}] Loading {fname} ...")
        arr = np.load(src, mmap_mode="r")

        # ensure numpy array
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr, dtype=DTYPE)
        else:
            arr = arr.astype(DTYPE, copy=False)

        print(f"   shape={arr.shape}, dtype={arr.dtype}, saving to {dst}")
        np.save(dst, arr) 

        # verify correctness
        test = np.load(dst, mmap_mode="r")
        print(f"verified: {test.shape}, {test.dtype}")

        if DELETE_ORIGINAL:
            os.remove(src)
            print(f"   deleted {fname}")

        del arr, test
        gc.collect()

    except Exception as e:
        print(f"failed for {fname}: {e}")

print("Conversion complete.")
