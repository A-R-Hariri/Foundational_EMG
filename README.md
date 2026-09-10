# Foundational EMG

Self-supervised pretraining of a shared EMG encoder across many datasets, recording devices, channel counts, sampling rates, and window lengths. The pretrained encoder is used as an initialization for downstream tasks such as gesture classification, with no or minimal labeled data from the target user or device.

This repository is the scale-up of the zero-shot cross-user work in [`EPN612_Cross_User`](https://github.com/A-R-Hariri/EPN612_Cross_User), moving from single-dataset cross-user generalization to representations learned over the full set of publicly available EMG datasets.

---

## Objectives

1. Learn a single encoder that accepts variable-length, variable-channel EMG without per-dataset retraining.
2. Train that encoder with objectives that require no gesture labels.
3. Provide a supervised evaluation path on EPN612 that is disjoint from the pretraining data.

Two self-supervised objectives are optimized jointly:

**VICReg** (variance-invariance-covariance regularization). Two independently augmented views of the same window are encoded and pooled. An invariance term matches the two embeddings, a variance term keeps per-dimension standard deviation above a target, and a covariance term decorrelates the embedding dimensions.

**MAE** (masked autoencoding). A fraction of the valid timesteps is replaced with a learned mask token after the convolutional stem. The decoder projects the encoded sequence back to channel space and reconstructs the signal at the masked positions only, on amplitude-normalized targets.

---

## Training configurations

| Script | Model | Input | Objectives | Parallelism | Batch |
|---|---|---|---|---|---|
| `pretrain_fsdp.py` | `EMGPretrainer` | variable length, variable channels | VICReg + MAE | FSDP, full shard | 128 |
| `pretrain_ddp.py` | `EMGTransformer` | fixed shape, single window size | VICReg | DDP | 2048 |

The FSDP configuration is the main path and consumes every window size and channel count produced by the preprocessing stage. The DDP configuration operates on one window size for fixed-shape batching and higher throughput.

---

## Repository structure

```
Foundational_EMG/
    models.py                   EMGPretrainer (FSDP path), EMGTransformer (DDP path)
    utils.py                    config constants, datasets, collation, loaders,
                                augmentation, losses, checkpoint I/O

    pretrain_fsdp.py            FSDP pretraining, VICReg + MAE, variable-length input
    pretrain_ddp.py             DDP pretraining, VICReg, fixed-shape input

    process_cross_dataset.py    parallel windowing of all LibEMG datasets
    process_epn612.py           EPN612 windowing and active-segment extraction

    Datasets/
        EPN612.py               EMGEPN612 LibEMG dataset class, JSON to HDF5 conversion
        EPN100.py               EMGEPN100 LibEMG dataset class, MAT to HDF5 conversion,
                                dual-device (Myo, gForce) with per-record device and
                                sampling-rate labels

    checkpoints/                written at runtime
    data_pickles/               written at runtime, cross-dataset SSL windows
    pickles/                    written at runtime, EPN612 supervised windows
```

---

## Models

### `EMGPretrainer` (FSDP path)

```
(B, L, C) padded input + time mask + channel mask + per-sample timestamps
    -> channel masking
    -> Conv1d stem, kernel 8, stride 4, padding 4, C_max -> 256, run in fp32
    -> sinusoidal time encoding built from real timestamps in seconds,
       interpolated to the reduced sequence length and concatenated, then
       projected back to d_model
    -> optional replacement of masked positions with a learned mask token
    -> Transformer encoder, 4 layers, 4 heads, pre-norm, FFN 4x, dropout 0.1
    -> VICReg head: mask-aware mean pooling, then 2-layer MLP
    -> MAE decoder: linear projection to C_max, linear interpolation to L
```

Timestamps are derived from each sample's own sampling rate, so datasets recorded at different rates place their tokens on a common physical time axis. Padding is tracked through the stem by max-pooling the time mask with the same kernel, stride, and padding, and the resulting mask is passed to the Transformer as a key padding mask.

Default size: `d_model=256`, 4 heads, 4 layers.

### `EMGTransformer` (DDP path)

```
(B, L, C) fixed-shape input
    -> trailing zero-padding trimmed per batch
    -> Conv1d 8 -> 128 (kernel 4) -> Conv1d 128 -> 128 (kernel 2), same padding
    -> LayerNorm
    -> learned positional embedding
    -> Transformer encoder, 4 layers, 2 heads, pre-norm, FFN 2x, dropout 0.1
    -> mean pooling over time, linear head
```

Default size: `d_model=128`, 2 heads, 4 layers.

---

## Data

### Cross-dataset windows for pretraining

```bash
python process_cross_dataset.py
```

Iterates over every dataset exposed by `libemg.datasets.get_dataset_list("ALL", True)` using a joblib process pool (up to 8 workers). For each dataset, sliding windows are extracted at 0.1, 0.25, 0.5, and 1.0 s with 50 percent overlap, using the dataset's own sampling rate to convert seconds to samples. Windows are stored as float16 and written to `data_pickles/` in chunks capped at 50 GB:

```
{dataset}_ws{tag}_chunk{n}.npy      shape (N, ws_samples, C)
```

where `tag` encodes the window duration, for example `0_1` for 0.1 s and `1_0` for 1.0 s. The window size is recovered from the filename at load time and combined with the array length to infer the sampling rate of each file.

For `EMGEPN612`, only subjects 276 to 305 of the training split are windowed. The remaining training subjects are reserved for the supervised cross-user experiments, keeping pretraining and supervised evaluation disjoint.

### EPN612 windows for supervised evaluation

```bash
python process_epn612.py
```

Builds the fixed cross-user split used by the downstream classification pipeline:

| Split | Subjects | Count |
|---|---|---|
| Train | 1 to 306 (trainingJSON) | 306 |
| Val | 307 to 332 | 26 |
| Test | 333 to 612 | 280 |

Windows are 40 samples with a 2-sample increment over the five static classes. Both raw and active-segment versions are written to `pickles/`. Active segments are located by smoothing the per-sample channel energy, normalizing it, and keeping the span above a 0.25 threshold; the rest-class repetitions are kept in full. Segment boundaries are saved alongside the windows.

### Dataset classes

`Datasets/EPN612.py` converts the released per-user JSON files into one HDF5 file per subject, holding EMG, gesture label, subject id, repetition index, and the ground-truth onset and offset indices. `EMGEPN612` then exposes the data as a LibEMG `OfflineDataHandler` with optional onset-based segmentation and optional relabeling of the pre-onset and post-offset regions.

`Datasets/EPN100.py` performs the equivalent conversion from MATLAB `.mat` files for the EMG-EPN-100 dataset, which records 12 gestures with two devices: the Myo armband at 200 Hz and 8-bit resolution and the gForce armband at 500 Hz and 12-bit resolution. `EMGEPN100` attaches a device id and a sampling rate to every record, giving explicit device labels for cross-device analysis.

---

## Training

### FSDP

```bash
torchrun --nproc_per_node=<N> pretrain_fsdp.py

python pretrain_fsdp.py            # single process
```

`NPYWindows` memory-maps every `.npy` file under `data_pickles/`, indexes them into one global window index, and reports the maximum channel count across the corpus. `collate_variable` pads each batch to the longest window and to the global channel maximum, and returns time masks, channel masks, and per-sample sampling rates. Splitting is 90/10 train/val with a fixed seed, and `DistributedSampler` is applied when running under `torchrun`.

Sharding uses `FULL_SHARD` with a module wrap policy over `Encoder`, `VICHead`, `MAEDecoder`, and `TransformerEncoderLayer`. Activation checkpointing is applied to every Transformer layer. Mixed precision keeps parameters and buffers in fp32 and reduces gradients in bf16. Gradients are clipped to a norm of 1.0. Non-finite losses are detected per step and the step is skipped with the gradients cleared.

Augmentation runs on GPU over the padded batch and respects both masks:

| Transform | Probability |
|---|---|
| Global amplitude scaling | 0.70 |
| Per-channel amplitude scaling | 0.50 |
| Baseline drift, linear or sinusoidal | 0.30 |
| Circular temporal shift within valid length | 0.70 |
| Time warping | 0.50 |
| Magnitude warping | 0.50 |
| Additive noise scaled to sample std | 0.60 |
| Channel dropout | 0.20 |
| Channel permutation | 0.50 |
| Moving-average lowpass | 0.30 |

Checkpoints are written to `checkpoints/fsdp_latest.pt` every epoch and `checkpoints/fsdp_best.pt` on validation improvement. Both store model, optimizer, scheduler, scaler, epoch, best validation loss, and the global channel maximum. Training resumes from the best checkpoint automatically when it exists.

### DDP

```bash
torchrun --nproc_per_node=<N> pretrain_ddp.py

python pretrain_ddp.py             # single process
```

`FixedWindowNPY` loads only the files matching `TARGET_WIN_SEC` (0.2 s by default), shuffles the global index with a fixed seed, and splits 90/10. Batches are fixed shape, so no custom collation is needed. TF32 matmul and `static_graph` DDP are enabled. Augmentation is the lighter fixed-shape variant: amplitude scaling, temporal roll, additive noise, and channel dropout.

Checkpoints: `checkpoints/ddp_latest.pt` and `checkpoints/ddp_best.pt`.

---

## Configuration

All shared constants live at the top of `utils.py`.

| Constant | Value | Scope |
|---|---|---|
| `DATA_PATH` | `data_pickles` | both |
| `CKPT_PATH` | `checkpoints` | both |
| `BATCH_SIZE` | 128 | FSDP |
| `DDP_BATCH_SIZE` | 2048 | DDP |
| `EPOCHS` | 200 | both |
| `LR` / `MIN_LR` | 1e-4 / 1e-6 | both |
| `LR_FACTOR` / `LR_PATIENCE` | 0.8 / 2 | both |
| `PATIENCE` | 10 | both |
| `SEED` | 67 | both |
| `LATENT_DIM` | 256 | FSDP |
| `NUM_HEADS` / `NUM_LAYERS` | 4 / 4 | FSDP |
| `CONV_KERNEL` / `CONV_STRIDE` | 8 / 4 | FSDP |
| `MAE_MASK_FRAC` | 0.3 | FSDP |
| `VIC_LOSS_W` / `MAE_LOSS_W` | 1.0 / 1.0 | FSDP |
| `TARGET_WIN_SEC` / `TARGET_FS` | 0.2 s / 500 Hz | DDP |

Both scripts use Adam with `ReduceLROnPlateau` on validation loss and stop early after `PATIENCE` epochs without improvement.

---

## Requirements

```bash
pip install torch numpy scipy scikit-learn joblib tqdm h5py libemg
```

Multi-GPU training uses `torchrun`, included with PyTorch. FSDP requires PyTorch 2.0 or newer.

---

## License

MIT. See `LICENSE`.

## Author

Amir Hariri, Institute of Biomedical Engineering, University of New Brunswick.