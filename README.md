# Foundational EMG

Self-supervised EMG pretraining across multiple datasets and device types, with cross-device generalization as the primary objective. The pretrained encoder can be fine-tuned for downstream tasks such as gesture classification with no or minimal labeled data.

This is the scale-up extension of the zero-shot cross-user EPN612 work, targeting foundational representations rather than single-dataset generalization.

---

## Overview

Two self-supervised objectives are used jointly during pretraining:

**VICReg** (Variance-Invariance-Covariance Regularization): two augmented views of each window are encoded and their embeddings are matched via invariance, while variance and covariance regularization prevent collapse.

**MAE** (Masked Autoencoder): a fraction of the encoded sequence is masked before the Transformer, and the decoder reconstructs the original signal at those positions.

Two training configurations are provided:

| Script | Model | Input | Objective | Parallelism |
|--------|-------|-------|-----------|-------------|
| `pretrain_fsdp.py` | `EMGPretrainer` | variable-length, variable-channel | VICReg + MAE | FSDP |
| `pretrain_ddp.py` | `EMGTransformer` | fixed-shape | VICReg | DDP |

The FSDP configuration is the production path. The DDP configuration is a simpler fixed-shape baseline.

---

## Repository Structure

```
Foundational_EMG/
    models.py                   -- EMGPretrainer (FSDP) and EMGTransformer (DDP)
    utils.py                    -- config, datasets, loaders, augmentation, losses, checkpointing

    pretrain_fsdp.py            -- FSDP pretraining: VICReg + MAE, variable-length input
    pretrain_ddp.py             -- DDP pretraining: VICReg only, fixed-shape input

    process_cross_dataset.py    -- parallel preprocessing for all LibEMG datasets
    process_epn612.py           -- EPN612-specific preprocessing for fine-tuning evaluation

    Datasets/                   -- LibEMG-compatible dataset classes (EPN612, etc.)
    checkpoints/                -- saved model checkpoints (created at runtime)
    data_pickles/               -- preprocessed .npy window files (created at runtime)
    pickles/                    -- EPN612 fine-tuning windows (created at runtime)
```

---

## Data

### Cross-dataset preprocessing (SSL pretraining)

```bash
python process_cross_dataset.py
```

Reads all datasets available through LibEMG and writes chunked float16 `.npy` files to `data_pickles/`. Each file stores windows of a single shape `(N, ws_samples, C)`. Multiple window sizes are produced per dataset (0.1, 0.25, 0.5, 1.0 s).

For EMGEPN612, only users 276-305 (30 users from the training split) are included to avoid overlap with the supervised cross-user evaluation.

### EPN612 fine-tuning preprocessing

```bash
python process_epn612.py
```

Produces windowed train/val/test splits from EPN612 (40-sample windows, 2-sample stride) in `pickles/`, in the format expected by downstream supervised evaluation.

---

## Training

### FSDP pretraining (recommended)

```bash
torchrun --nproc_per_node=<N> pretrain_fsdp.py
# single GPU
python pretrain_fsdp.py
```

Loads all files from `data_pickles/` via memory-mapped `NPYWindows`. Handles variable channel counts and lengths across datasets via padded collation and channel masks. Applies a rich augmentation pipeline (amplitude scaling, baseline drift, temporal shift, time warping, magnitude warping, noise, channel dropout, channel permutation, lowpass smoothing).

Checkpoints are written to `checkpoints/fsdp_best.pt` and `checkpoints/fsdp_latest.pt`. Resumes automatically if a checkpoint exists.

### DDP pretraining

```bash
torchrun --nproc_per_node=<N> pretrain_ddp.py
# single GPU
python pretrain_ddp.py
```

Loads only files matching the configured window size (`TARGET_WIN_SEC = 0.2 s` by default) for fixed-shape batching. Simpler augmentation; no MAE objective.

Checkpoints: `checkpoints/ddp_best.pt`, `checkpoints/ddp_latest.pt`.

---

## Dependencies

```
torch
numpy
scipy
scikit-learn
joblib
tqdm
libemg
```

```bash
pip install torch numpy scipy scikit-learn joblib tqdm libemg
```

FSDP and DDP training require `torchrun` (included with PyTorch >= 1.10).

---

## Author

Amir Hariri, Institute of Biomedical Engineering, University of New Brunswick.
