"""
EPN612 preprocessing for supervised fine-tuning / evaluation.

Produces windowed train/val/test splits from the EPN612 dataset in the format
used by the supervised cross-user evaluation pipeline. This script is a companion
to the SSL pretraining work; it is not part of the SSL data pipeline (which
uses process_cross_dataset.py instead).

Fixed split (never randomized):
    Train : users 1-306  (306 users, from trainingJSON)
    Val   : users 307-332 (26 users)
    Test  : users 333-612 (280 users)

Run:
    python process_epn612.py
"""

import gc
import copy
import os
import numpy as np
from os.path import join

from Datasets.EPN612 import EMGEPN612

# ======== CONFIG ========

PICKLE_PATH = "pickles"
DTYPE = np.float16
SEQ = 40
INC = 2
VAL_CUTOFF = 332
CH = 8

os.makedirs(PICKLE_PATH, exist_ok=True)

# ======== DATA ========

dataset = EMGEPN612()
data = dataset.prepare_data(split=True, segment=True, relabel_seg=0)

train_data = data["Train"].isolate_data("classes", [0, 1, 2, 3, 4], fast=True)
test_data = data["Test"].isolate_data("base_class", [0, 1, 2, 3, 4], fast=True)
val_data = test_data.isolate_data("subjects", list(range(306, VAL_CUTOFF)), fast=True)
test_data = test_data.isolate_data("subjects", list(range(VAL_CUTOFF, 612)), fast=True)

np.save(join(PICKLE_PATH, "train_data"), train_data)
np.save(join(PICKLE_PATH, "val_data"), val_data)
np.save(join(PICKLE_PATH, "test_data"), test_data)

# -------- raw windows --------

train_windows, train_meta = train_data.parse_windows(SEQ, INC)
np.save(join(PICKLE_PATH, "train_windows"), train_windows.astype(DTYPE))
np.save(join(PICKLE_PATH, "train_meta"), train_meta)
del train_windows
gc.collect()

val_windows, val_meta = val_data.parse_windows(SEQ, INC)
np.save(join(PICKLE_PATH, "val_windows"), val_windows.astype(DTYPE))
np.save(join(PICKLE_PATH, "val_meta"), val_meta)
del val_windows
gc.collect()

test_windows, test_meta = test_data.parse_windows(SEQ, INC)
np.save(join(PICKLE_PATH, "test_windows"), test_windows.astype(DTYPE))
np.save(join(PICKLE_PATH, "test_meta"), test_meta)
del test_windows
gc.collect()

# -------- active segment extraction --------

def _energy_signal(data_i, window_size=5):
    energy = np.sum(data_i ** 2, axis=1)
    return np.convolve(energy, np.ones(window_size) / window_size, mode="same")

def extract_active_segments(odh, threshold=0.25, n_min=SEQ + INC):
    """Replace each repetition with its energy-gated active portion."""
    segmented = copy.deepcopy(odh)
    bounds = []
    total_orig = total_kept = 0
    for i in range(len(odh.data)):
        d = np.asarray(odh.data[i])
        T = d.shape[0]
        total_orig += T
        cls = int(odh.classes[i][0])
        if cls == 0:
            segmented.data[i] = d
            bounds.append((0, T))
            total_kept += T
            continue
        sig = _energy_signal(d)
        sig = (sig - sig.min()) / (sig.max() - sig.min() + 1e-8)
        active = np.where(sig > threshold)[0]
        if len(active) > 1 and (active[-1] - active[0]) > n_min:
            start, end = int(active[0]), int(active[-1] + 1)
        else:
            start, end = 0, T
        segmented.data[i] = d[start:end]
        segmented.classes[i] = odh.classes[i][start:end]
        segmented.reps[i] = odh.reps[i][start:end]
        segmented.subjects[i] = odh.subjects[i][start:end]
        bounds.append((start, end))
        total_kept += end - start
    pct = 100 * (total_orig - total_kept) / total_orig
    print(f"Segmentation: removed {pct:.1f}% of samples.")
    return segmented, bounds

train_seg, train_bounds = extract_active_segments(train_data)
val_seg, val_bounds = extract_active_segments(val_data)
test_seg, test_bounds = extract_active_segments(test_data)

np.save(join(PICKLE_PATH, "train_segmented_bounds"), np.array(train_bounds))
np.save(join(PICKLE_PATH, "val_segmented_bounds"), np.array(val_bounds))
np.save(join(PICKLE_PATH, "test_segmented_bounds"), np.array(test_bounds))

train_wins_seg, train_meta_seg = train_seg.parse_windows(SEQ, INC)
val_wins_seg, val_meta_seg = val_seg.parse_windows(SEQ, INC)
test_wins_seg, test_meta_seg = test_seg.parse_windows(SEQ, INC)

np.save(join(PICKLE_PATH, "train_windows_segmented"), train_wins_seg.astype(DTYPE))
np.save(join(PICKLE_PATH, "train_meta_segmented"), train_meta_seg)
np.save(join(PICKLE_PATH, "val_windows_segmented"), val_wins_seg.astype(DTYPE))
np.save(join(PICKLE_PATH, "val_meta_segmented"), val_meta_seg)
np.save(join(PICKLE_PATH, "test_windows_segmented"), test_wins_seg.astype(DTYPE))
np.save(join(PICKLE_PATH, "test_meta_segmented"), test_meta_seg)

del train_seg, val_seg, test_seg
del train_wins_seg, val_wins_seg, test_wins_seg
gc.collect()

print("Done.")
