import gc
from os.path import join
import numpy as np 
import copy

import libemg
from libemg.datasets import get_dataset_list
from libemg.feature_extractor import FeatureExtractor
from libemg.offline_metrics import OfflineMetrics


from utils import *


# ======== DATA ========
from EPN612 import EMGEPN612
dataset = EMGEPN612()
data = dataset.prepare_data(split=True, segment=True, relabel_seg=0)

# ======== RAW ========
train_data = data['Train']
train_data = train_data.isolate_data("classes", [0, 1, 2, 3, 4], fast=True)

test_data = data['Test']
test_data = test_data.isolate_data("base_class", [0, 1, 2, 3, 4], fast=True)
val_data = test_data.isolate_data("subjects", list(range(306, VAL_CUTOFF)), fast=True)
test_data = test_data.isolate_data("subjects", list(range(VAL_CUTOFF, 612)), fast=True)

np.save(join(PICKLE_PATH, 'train_data'), train_data)
np.save(join(PICKLE_PATH, 'val_data'), val_data)
np.save(join(PICKLE_PATH, 'test_data'), test_data)

train_windows, train_meta = train_data.parse_windows(SEQ, INC)
np.save(join(PICKLE_PATH, 'train_windows'), train_windows.astype(DTYPE))
np.save(join(PICKLE_PATH, 'train_meta'), train_meta)
del train_windows
gc.collect()

val_windows, val_meta = val_data.parse_windows(SEQ, INC)
np.save(join(PICKLE_PATH, 'val_windows'), val_windows.astype(DTYPE))
np.save(join(PICKLE_PATH, 'val_meta'), val_meta)
del val_windows
gc.collect()

test_windows, test_meta = test_data.parse_windows(SEQ, INC)
np.save(join(PICKLE_PATH, 'test_windows'), test_windows.astype(DTYPE))
np.save(join(PICKLE_PATH, 'test_meta'), test_meta)
del test_windows
gc.collect()

# ======== SEGMENTED ========
train_data_segmented = copy.deepcopy(train_data)
val_data_segmented = copy.deepcopy(val_data)
test_data_segmented = copy.deepcopy(test_data)

del train_data
del val_data
del test_data
gc.collect()

def tkeo(x):
    return x[1:-1]**2 - x[:-2] * x[2:]

def extract_active_segment(emg_data, window_size=5, threshold=0.25, 
                        n_samples=SEQ + INC, method='energy'):
    segmented_data = []
    segmented_classes = []
    segmented_reps = []
    segmented_subjects = []
    segmented_bounds = []

    total_original = 0
    total_kept = 0

    for i in range(len(emg_data.data)):
        data_i = np.asarray(emg_data.data[i])
        class_i = np.asarray(emg_data.classes[i])
        rep_i = np.asarray(emg_data.reps[i])
        subj_i = np.asarray(emg_data.subjects[i])

        t, ch = data_i.shape
        assert ch == 8, f"Expected 8 channels, got {ch} at index {i}"
        total_original += t

        for meta_arr, name in zip([class_i, rep_i, subj_i], ['classes', 'reps', 'subjects']):
            if meta_arr.shape != (t, 1):
                raise ValueError(f"{name}[{i}] must have shape (t, 1), got {meta_arr.shape}")
            if not np.all(meta_arr == meta_arr[0]):
                raise ValueError(f"{name}[{i}] is not constant across time")

        current_class = class_i[0, 0]

        if method == 'tkeo':
            best_val = -np.inf
            signal = None
            for ch_idx in range(ch):
                x = data_i[:, ch_idx]
                if len(x) < 3:
                    continue
                e = tkeo(x)
                smoothed = np.convolve(e, np.ones(window_size) / window_size, mode='same')
                max_energy = np.max(smoothed)
                if max_energy > best_val:
                    best_val = max_energy
                    signal = smoothed
            if signal is None:
                signal = np.zeros(t - 2)
                signal = np.pad(signal, (1, 1), mode='constant')

        elif method == 'energy':
            ch_candidates = list(range(ch))

            channel_energies = [np.sum(data_i[:, idx] ** 2) for idx in ch_candidates]
            main_ch_idx = ch_candidates[np.argmax(channel_energies)]
            signal = data_i[:, main_ch_idx] ** 2
            signal = np.convolve(signal, np.ones(window_size) / window_size, mode='same')

        elif method == 'variance':
            if current_class == 3:
                ch_candidates = [3, 4]
            elif current_class == 4:
                ch_candidates = [0, 7]
            else:
                ch_candidates = list(range(ch))

            mavs = [np.var(np.abs(data_i[:, idx])) for idx in ch_candidates]
            main_ch_idx = ch_candidates[np.argmax(mavs)]
            signal = np.abs(data_i[:, main_ch_idx])
            signal = np.convolve(signal, np.ones(window_size) / window_size, mode='same')

        else:
            raise ValueError(f"Unknown method: {method}. Use 'tkeo', 'energy', or 'variance'.")

        signal_norm = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
        active_indices = np.where(signal_norm > threshold)[0]

        if len(active_indices) > 1:
            start_idx, end_idx = active_indices[0], active_indices[-1] + 1
        else:
            start_idx, end_idx = 0, 0

        if (end_idx - start_idx) <= n_samples or current_class == 0 or len(active_indices) == 0:
            start_idx, end_idx = 0, t

        segmented_data.append(data_i[start_idx:end_idx])
        segmented_classes.append(class_i[start_idx:end_idx])
        segmented_reps.append(rep_i[start_idx:end_idx])
        segmented_subjects.append(subj_i[start_idx:end_idx])
        segmented_bounds.append((start_idx, end_idx))
        total_kept += end_idx - start_idx

    percent_removed = 100 * (total_original - total_kept) / total_original
    print(f"Total data removed by segmenting: {percent_removed:.2f}%")

    return (
        segmented_data,
        segmented_classes,
        segmented_reps,
        segmented_subjects,
        segmented_bounds
    )

train_data_segmented.data, train_data_segmented.classes, train_data_segmented.reps, train_data_segmented.subjects, train_segmented_bounds = extract_active_segment(train_data_segmented)
val_data_segmented.data, val_data_segmented.classes, val_data_segmented.reps, val_data_segmented.subjects, val_segmented_bounds = extract_active_segment(val_data_segmented)
test_data_segmented.data, test_data_segmented.classes, test_data_segmented.reps, test_data_segmented.subjects, test_segmented_bounds = extract_active_segment(test_data_segmented)

train_windows_segmented, train_meta_segmented = train_data_segmented.parse_windows(SEQ, INC)
val_windows_segmented, val_meta_segmented = val_data_segmented.parse_windows(SEQ, INC)
test_windows_segmented, test_meta_segmented = test_data_segmented.parse_windows(SEQ, INC)

np.save(join(PICKLE_PATH, 'train_segmented_bounds'), np.array(train_segmented_bounds))
np.save(join(PICKLE_PATH, 'val_segmented_bounds'), np.array(val_segmented_bounds))
np.save(join(PICKLE_PATH, 'test_segmented_bounds'), np.array(test_segmented_bounds))

np.save(join(PICKLE_PATH, 'train_data_segmented'), train_data_segmented)
np.save(join(PICKLE_PATH, 'val_data_segmented'), val_data_segmented)
np.save(join(PICKLE_PATH, 'test_data_segmented'), test_data_segmented)

np.save(join(PICKLE_PATH, 'train_windows_segmented'), train_windows_segmented.astype(DTYPE))
np.save(join(PICKLE_PATH, 'train_meta_segmented'), train_meta_segmented)
np.save(join(PICKLE_PATH, 'val_windows_segmented'), val_windows_segmented.astype(DTYPE))
np.save(join(PICKLE_PATH, 'val_meta_segmented'), val_meta_segmented)
np.save(join(PICKLE_PATH, 'test_windows_segmented'), test_windows_segmented.astype(DTYPE))
np.save(join(PICKLE_PATH, 'test_meta_segmented'), test_meta_segmented)

del train_data_segmented
del val_data_segmented
del test_data_segmented
del train_windows_segmented
del train_meta_segmented
del val_windows_segmented
del val_meta_segmented
del test_windows_segmented
del test_meta_segmented
gc.collect()

# ======== RELABELED ========
train_data = np.load(join(PICKLE_PATH, 'train_data.npy'), allow_pickle=True).item()
val_data = np.load(join(PICKLE_PATH, 'val_data.npy'), allow_pickle=True).item()
test_data = np.load(join(PICKLE_PATH, 'test_data.npy'), allow_pickle=True).item()

train_segmented_bounds = np.load(join(PICKLE_PATH, 'train_segmented_bounds.npy'))
val_segmented_bounds = np.load(join(PICKLE_PATH, 'val_segmented_bounds.npy'))
test_segmented_bounds = np.load(join(PICKLE_PATH, 'test_segmented_bounds.npy'))

t_data = {'data': [], 'classes': [], 'subjects': []}
for i in range(len(train_data.data)):
    if train_data.classes[i][0].item() == 0:
        t_data['data'].append(train_data.data[i])
        t_data['classes'].append(train_data.classes[i][0].item())
        t_data['subjects'].append(train_data.subjects[i][0].item())
        continue
    t_data['data'].append(train_data.data[i][:train_segmented_bounds[i][0]])
    t_data['classes'].append(0)
    t_data['subjects'].append(train_data.subjects[i][0].item())
    t_data['data'].append(train_data.data[i][train_segmented_bounds[i][0]:train_segmented_bounds[i][1]])
    t_data['classes'].append(train_data.classes[i][0].item())
    t_data['subjects'].append(train_data.subjects[i][0].item())
    t_data['data'].append(train_data.data[i][train_segmented_bounds[i][1]:])
    t_data['classes'].append(0)
    t_data['subjects'].append(train_data.subjects[i][0].item())

v_data = {'data': [], 'classes': [], 'subjects': []}
for i in range(len(val_data.data)):
    if val_data.classes[i][0].item() == 0:
        v_data['data'].append(val_data.data[i])
        v_data['classes'].append(val_data.classes[i][0].item())
        v_data['subjects'].append(val_data.subjects[i][0].item())
        continue
    v_data['data'].append(val_data.data[i][:val_segmented_bounds[i][0]])
    v_data['classes'].append(0)
    v_data['subjects'].append(val_data.subjects[i][0].item())
    v_data['data'].append(val_data.data[i][val_segmented_bounds[i][0]:val_segmented_bounds[i][1]])
    v_data['classes'].append(val_data.classes[i][0].item())
    v_data['subjects'].append(val_data.subjects[i][0].item())
    v_data['data'].append(val_data.data[i][val_segmented_bounds[i][1]:])
    v_data['classes'].append(0)
    v_data['subjects'].append(val_data.subjects[i][0].item())

ts_data = {'data': [], 'classes': [], 'subjects': []}
for i in range(len(test_data.data)):
    if test_data.classes[i][0].item() == 0:
        ts_data['data'].append(test_data.data[i])
        ts_data['classes'].append(test_data.classes[i][0].item())
        ts_data['subjects'].append(test_data.subjects[i][0].item())
        continue
    ts_data['data'].append(test_data.data[i][:test_segmented_bounds[i][0]])
    ts_data['classes'].append(0)
    ts_data['subjects'].append(test_data.subjects[i][0].item())
    ts_data['data'].append(test_data.data[i][test_segmented_bounds[i][0]:test_segmented_bounds[i][1]])
    ts_data['classes'].append(test_data.classes[i][0].item())
    ts_data['subjects'].append(test_data.subjects[i][0].item())
    ts_data['data'].append(test_data.data[i][test_segmented_bounds[i][1]:])
    ts_data['classes'].append(0)
    ts_data['subjects'].append(test_data.subjects[i][0].item())

def window_dataset(data, window, stride) :

    data, gestures, subjects = data['data'], data['classes'], data['subjects']

    X_chunks = []
    y_gesture = []
    y_subject = []
    y_rep = []

    for i in range(len(data)):
        # if i % 10000 == 0:
        #     print(i / len(data))
        win_len = window
        stride_len = stride

        T = data[i].shape[0]

        if T < window:
            continue

        starts = np.arange(0, T - win_len + 1, stride_len)
        idx = starts[:, None] + np.arange(win_len)[None, :]

        windows = data[i][idx]  # [n_win, win_len, 8]
        windows = windows.reshape((-1, CH, SEQ))

        n_win = windows.shape[0]

        X_chunks.append(windows.astype(np.float32))
        y_gesture.append(np.full(n_win, gestures[i], dtype=np.int64))
        y_subject.append(np.full(n_win, subjects[i], dtype=np.int64))
        # y_rep.append(np.full(n_win, train_data.reps[i][0].item(), dtype=np.int64))

    X = X_chunks
    meta = {
        "classes": np.concatenate(y_gesture),
        "subjects": np.concatenate(y_subject),}
        # "rep": np.concatenate(y_rep),}

    return X, meta

X, meta = window_dataset(t_data, SEQ, INC)
X = np.concatenate(X, axis=0) 
y = meta

np.save(join(PICKLE_PATH, 'train_windows_relabeled'), X.astype(DTYPE))
np.save(join(PICKLE_PATH, 'train_meta_relabeled'), y)

del X, y
gc.collect()

X_v, meta = window_dataset(v_data, SEQ, INC)
X_v = np.concatenate(X_v, axis=0) 
y_v = meta

X_t, meta = window_dataset(ts_data, SEQ, INC)
X_t = np.concatenate(X_t, axis=0) 
y_t = meta

np.save(join(PICKLE_PATH, 'val_windows_relabeled'), X_v.astype(DTYPE))
np.save(join(PICKLE_PATH, 'val_meta_relabeled'), y_v)
np.save(join(PICKLE_PATH, 'test_windows_relabeled'), X_t.astype(DTYPE))
np.save(join(PICKLE_PATH, 'test_meta_relabeled'), y_t)


print('DONE.')