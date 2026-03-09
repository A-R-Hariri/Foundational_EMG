import warnings, os, gc, math, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from joblib import Parallel, delayed
from tqdm import tqdm

from libemg.datasets import get_dataset_list

# =================== CONFIG ===================
SAVE_PATH    = "data_pickles"
os.makedirs(SAVE_PATH, exist_ok=True)

WIN_RANGE    = [0.1, 0.25, 0.5, 1.0]   # seconds
STRIDE_FRAC  = 0.5
DTYPE        = np.float16
MAX_BYTES    = 50 * (1024 ** 3)        # 50 GB per chunk
N_JOBS       = max(1, min(os.cpu_count() or 1, 8))
VERBOSE      = 10                   
# ===============================================

def safe_ws_str(ws: float) -> str:
    s = f"{ws}"
    return s.replace('.', '_')

def estimate_chunk_capacity(ws_samples: int, ch: int) -> int:
    """How many windows fit under MAX_BYTES for given shape."""
    bytes_per_window = ws_samples * ch * np.dtype(DTYPE).itemsize
    if bytes_per_window <= 0:
        return 0
    return int(MAX_BYTES // bytes_per_window)

def make_windows(sig: np.ndarray, ws_samples: int, stride_samples: int) -> np.ndarray:
    """
    Vectorized window extraction.
    - sig: (T, C)
    - returns: (Nw, ws_samples, C) with end padding only if T < ws_samples
    """
    T, C = sig.shape
    if T < ws_samples:
        pad = np.zeros((ws_samples - T, C), dtype=sig.dtype)
        sig = np.concatenate([sig, pad], axis=0)
        T = sig.shape[0]

    # sliding_window_view over (time, channel) with window (ws_samples, C)
    # result shape before slicing: (T - ws_samples + 1, 1, ws_samples, C)
    wins = sliding_window_view(sig, (ws_samples, C))
    wins = wins[:, 0, :, :]  # drop the singleton dimension
    # stride in time
    if stride_samples > 1:
        wins = wins[::stride_samples]
    return wins  # (Nw, ws_samples, C)

def save_chunk(array3d: np.ndarray, out_path: str):
    np.save(out_path, array3d)

def process_one_dataset(name: str):
    """
    Process a single dataset end-to-end, writing chunked .npy files per window size.
    """
    gc.disable()
    try:
        ds_cls = get_dataset_list("ALL", True)[name]
        ds = ds_cls()
        fs = getattr(ds, "sampling", None)
        ch = getattr(ds, 'num_channels', None)

        # EMGEPN612 test set untouched
        if name == "EMGEPN612":
            data = ds.prepare_data(split=True)["Train"] \
                     .isolate_data("subjects", list(range(306 - 30, 306)), fast=True).data
            print(f"[{name}] Using TRAIN split only.")
        else:
            data = ds.prepare_data(split=False).data  # full dataset

        if len(data) == 0:
            print(f"[{name}] Empty dataset, skipping.")
            return
        ch = int(data[0].shape[1])
        if fs is None or ch is None:
            raise RuntimeError(f"[{name}] fs or ch is None.")

        print(f"\n=== {name} | fs={fs} Hz | channels={ch} | samples={len(data)} ===")

        # skip samples with mismatched channel count (no channel changes allowed)
        valid_idx = [i for i, x in enumerate(data) if x.shape[1] == ch]
        if len(valid_idx) != len(data):
            print(f"[{name}] Warning: {len(data) - len(valid_idx)} samples skipped due to channel mismatch.")

        for ws in WIN_RANGE:
            ws_samples = int(round(ws * fs))
            if ws_samples <= 0:
                print(f"[{name}] ws={ws} -> ws_samples={ws_samples} invalid, skipping.")
                continue
            stride_samples = max(1, int(round(STRIDE_FRAC * ws_samples)))

            max_N = estimate_chunk_capacity(ws_samples, ch)
            if max_N <= 0:
                print(f"[{name}] ws={ws}: windows too large for MAX_BYTES={MAX_BYTES}, skipping.")
                continue

            ws_tag = safe_ws_str(ws)
            chunk_idx = 0
            # preallocate one chunk buffer
            X = np.zeros((max_N, ws_samples, ch), dtype=DTYPE)
            idx = 0

            pbar = tqdm(valid_idx, desc=f"{name} [ws={ws}s]", miniters=1, smoothing=0.1)
            for i in pbar:
                sig = data[i]
                # typecast once
                if sig.dtype != DTYPE:
                    sig = sig.astype(DTYPE, copy=False)

                wins = make_windows(sig, ws_samples, stride_samples)  # (nwin, T, C)
                nwin = wins.shape[0]
                # fast fill in chunks
                pos = 0
                while pos < nwin:
                    space = max_N - idx
                    if space == 0:
                        # dump full chunk
                        out_name = os.path.join(SAVE_PATH, f"{name}_ws{ws_tag}_chunk{chunk_idx}.npy")
                        save_chunk(X, out_name)
                        print(f"[{name}] saved {out_name} ({X.shape}, {X.nbytes/1e9:.2f} GB)")
                        # reallocate chunk buffer
                        del X
                        gc.collect()
                        X = np.zeros((max_N, ws_samples, ch), dtype=DTYPE)
                        idx = 0
                        chunk_idx += 1
                        space = max_N
                    take = min(space, nwin - pos)
                    X[idx:idx+take] = wins[pos:pos+take]
                    idx += take
                    pos += take

            # save tail
            if idx > 0:
                out_name = os.path.join(SAVE_PATH, f"{name}_ws{ws_tag}_chunk{chunk_idx}.npy")
                save_chunk(X[:idx], out_name)
                print(f"[{name}] saved {out_name} (({idx}, {ws_samples}, {ch}), {(X[:idx].nbytes)/1e9:.2f} GB)")
            # cleanup for this ws
            del X
            gc.collect()

        # explicit cleanup
        del data, ds
        gc.collect()
        print(f"[{name}] done.")

    except Exception as e:
        print(f"[{name}] Skipped due to error: {e}")
    finally:
        gc.enable()

def main():
    keys = list(get_dataset_list("ALL", True).keys())
    print(f"Datasets to process: {len(keys)} | Parallel workers: {N_JOBS}")
    # Parallelize per dataset
    Parallel(n_jobs=N_JOBS, backend="multiprocessing", verbose=0)(
        delayed(process_one_dataset)(k) for k in keys
    )
    print("\nAll datasets processed.")

if __name__ == "__main__":
    main()
