"""
Cross-dataset EMG preprocessing pipeline.

Iterates over all LibEMG datasets in parallel, extracts sliding windows at multiple
window sizes, and writes chunked float16 .npy files to data_pickles/.

Each output file is named:
    {dataset}_{ws_tag}_chunk{n}.npy   shape (N, ws_samples, C)

where ws_tag encodes the window duration (e.g., 0_1 for 0.1 s, 1_0 for 1.0 s).

For EMGEPN612, only the training-split subjects listed in EPN612_TRAIN_SUBJECTS are
used, to avoid data leakage with the supervised EPN612 cross-user experiments.

Run:
    python process_cross_dataset.py
"""

import os
import gc
import warnings
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from joblib import Parallel, delayed
from tqdm import tqdm
from libemg.datasets import get_dataset_list

os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")

# ======== CONFIG ========

SAVE_PATH = "data_pickles"
WIN_SIZES = [0.1, 0.25, 0.5, 1.0]
STRIDE_FRAC = 0.5
DTYPE = np.float16
MAX_BYTES = 50 * (1024 ** 3)
N_JOBS = max(1, min(os.cpu_count() or 1, 8))

# EPN612 training subjects used for SSL. The remaining 276 training subjects are
# held exclusively for the supervised cross-user classification experiments.
EPN612_TRAIN_SUBJECTS = list(range(276, 306))

# ======== WINDOWING ========

def _ws_tag(ws):
    return str(ws).replace(".", "_")

def _chunk_capacity(ws_samples, ch):
    bpw = ws_samples * ch * np.dtype(DTYPE).itemsize
    return int(MAX_BYTES // bpw) if bpw > 0 else 0

def _make_windows(sig, ws_samples, stride_samples):
    T, C = sig.shape
    if T < ws_samples:
        sig = np.concatenate([sig, np.zeros((ws_samples - T, C), dtype=sig.dtype)], axis=0)
    wins = sliding_window_view(sig, (ws_samples, C))[:, 0, :, :]
    if stride_samples > 1:
        wins = wins[::stride_samples]
    return wins

# ======== PER-DATASET PROCESSOR ========

def process_dataset(name):
    gc.disable()
    try:
        ds_cls = get_dataset_list("ALL", True)[name]
        ds = ds_cls()
        fs = getattr(ds, "sampling", None)

        if name == "EMGEPN612":
            data = (
                ds.prepare_data(split=True)["Train"]
                .isolate_data("subjects", EPN612_TRAIN_SUBJECTS, fast=True)
                .data
            )
        else:
            data = ds.prepare_data(split=False).data

        if not data:
            print(f"[{name}] empty, skipping.")
            return

        ch = int(data[0].shape[1])
        if fs is None:
            raise RuntimeError(f"[{name}] sampling rate unknown.")

        valid_idx = [i for i, x in enumerate(data) if x.shape[1] == ch]
        n_skipped = len(data) - len(valid_idx)
        if n_skipped > 0:
            print(f"[{name}] skipped {n_skipped} samples with mismatched channels.")

        print(f"[{name}] fs={fs} ch={ch} n={len(valid_idx)}")

        for ws in WIN_SIZES:
            ws_samples = int(round(ws * fs))
            if ws_samples <= 0:
                continue
            stride_samples = max(1, int(round(STRIDE_FRAC * ws_samples)))
            max_N = _chunk_capacity(ws_samples, ch)
            if max_N <= 0:
                print(f"[{name}] ws={ws}: exceeds MAX_BYTES, skipping.")
                continue

            tag = _ws_tag(ws)
            chunk_idx = 0
            X = np.zeros((max_N, ws_samples, ch), dtype=DTYPE)
            ptr = 0

            for i in tqdm(valid_idx, desc=f"{name} ws={ws}s", miniters=1, leave=False):
                sig = data[i].astype(DTYPE, copy=False)
                wins = _make_windows(sig, ws_samples, stride_samples)
                nwin = wins.shape[0]
                pos = 0
                while pos < nwin:
                    space = max_N - ptr
                    if space == 0:
                        out = os.path.join(SAVE_PATH, f"{name}_ws{tag}_chunk{chunk_idx}.npy")
                        np.save(out, X)
                        print(f"[{name}] {out}")
                        del X
                        gc.collect()
                        X = np.zeros((max_N, ws_samples, ch), dtype=DTYPE)
                        ptr = 0
                        chunk_idx += 1
                        space = max_N
                    take = min(space, nwin - pos)
                    X[ptr:ptr + take] = wins[pos:pos + take]
                    ptr += take
                    pos += take

            if ptr > 0:
                out = os.path.join(SAVE_PATH, f"{name}_ws{tag}_chunk{chunk_idx}.npy")
                np.save(out, X[:ptr])
                print(f"[{name}] {out}")

            del X
            gc.collect()

        del data, ds
        gc.collect()
        print(f"[{name}] done.")

    except Exception as e:
        print(f"[{name}] failed: {e}")
    finally:
        gc.enable()


def main():
    os.makedirs(SAVE_PATH, exist_ok=True)
    keys = list(get_dataset_list("ALL", True).keys())
    print(f"{len(keys)} datasets | {N_JOBS} workers")
    Parallel(n_jobs=N_JOBS, backend="multiprocessing", verbose=0)(
        delayed(process_dataset)(k) for k in keys
    )
    print("Done.")


if __name__ == "__main__":
    main()
