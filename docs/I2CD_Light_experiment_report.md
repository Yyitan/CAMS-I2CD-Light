# CAMS-I2CD-Light Experiment Report

## Version v1: I2CD-light MVP

### Motivation

This version keeps the original CAMS inference path unchanged and adds an auxiliary training objective inspired by I2CD:

1. Disentangle the CAMS composition image feature into attribute and object features.
2. Compose the predicted attribute/object features back into a composition feature.
3. Re-disentangle the recomposed feature and enforce semantic and cycle consistency.

The goal of this MVP is to verify whether the disentangle-compose-redisentangle constraint can regularize CAMS features before changing test-time scoring.

### Code Changes

- Added `model/i2cd_light.py`.
- Added an `I2CDLight` auxiliary head in `model/cams.py`.
- Added auxiliary losses:
  - `lambda_i2cd_dis`: predicted attribute/object CE loss.
  - `lambda_i2cd_comp`: recomposed pair CE loss.
  - `lambda_i2cd_re`: re-disentangled attribute/object CE loss.
  - `lambda_i2cd_cycle`: feature-level cycle consistency.
  - `lambda_i2cd_anchor`: alignment to CAMS attribute/object branch features.
- Updated `model/configure_model.py` so parameters with `i2cd` in their name are trainable.
- Updated `train.py` to create `save_path` automatically and show lightweight I2CD loss diagnostics in the progress bar.
- Added configs:
  - `configs/utzappos_i2cd_light.yml`
  - `configs/mit_i2cd_light.yml`

### Default Loss Weights

```text
lambda_i2cd_dis    = 0.10
lambda_i2cd_comp   = 0.10
lambda_i2cd_re     = 0.05
lambda_i2cd_cycle  = 0.05
lambda_i2cd_anchor = 0.05
```

### Experiment Commands

UT-Zappos:

```bash
cd ./lyt/czsl/CAMS-I2CD-Light
source ./anaconda3/etc/profile.d/conda.sh
conda activate crdiff
mkdir -p output/utzappos_i2cd_light

CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --config configs/utzappos_i2cd_light.yml \
  --dataset_path data/ut-zap50k \
  2>&1 | tee output/utzappos_i2cd_light/train.log
```

MIT-States:

```bash
cd ./lyt/czsl/CAMS-I2CD-Light
source ./anaconda3/etc/profile.d/conda.sh
conda activate crdiff
mkdir -p output/mit_i2cd_light

CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --config configs/mit_i2cd_light.yml \
  --dataset_path data/mit-states \
  2>&1 | tee output/mit_i2cd_light/train.log
```

### Results

UT-Zappos seed 0, closed-world test with `val_best.pt`:

```text
CAMS baseline:
S 68.62 | U 76.10 | HM 56.56 | AUC 43.76

I2CD-light v1:
S 69.01 | U 73.88 | HM 54.79 | AUC 42.93
```

UT-Zappos seed 0, open-world test with `final_model.pt`:

```text
CAMS baseline:
S 68.62 | U 64.83 | HM 52.13 | AUC 36.25

I2CD-light v1:
S 69.01 | U 63.67 | HM 50.83 | AUC 36.60
```

### Analysis

The v1 auxiliary head is trainable and the losses decrease, but it does not produce a stable gain. Closed-world AUC drops by 0.83 and open-world AUC only improves by 0.35 while unseen/HM drop. This suggests the auxiliary I2CD constraint is active but likely too strong or not aligned enough with the original CAMS representation.

## Version v1a: Lower I2CD Loss Weights

### Motivation

The first controlled follow-up keeps the v1 architecture unchanged and reduces the auxiliary loss weights by 5x. This tests whether v1 failed because the I2CD-light losses over-regularized CAMS features.

### Changes Compared With v1

```text
lambda_i2cd_dis:    0.10 -> 0.02
lambda_i2cd_comp:   0.10 -> 0.02
lambda_i2cd_re:     0.05 -> 0.01
lambda_i2cd_cycle:  0.05 -> 0.01
lambda_i2cd_anchor: 0.05 -> 0.01
```

Config:

```text
configs/utzappos_i2cd_light_v1a.yml
```

### Experiment Command

```bash
cd ./lyt/czsl/CAMS-I2CD-Light
source ./anaconda3/etc/profile.d/conda.sh
conda activate crdiff
mkdir -p output/utzappos_i2cd_light_v1a

CUDA_VISIBLE_DEVICES=3 python -u train.py \
  --config configs/utzappos_i2cd_light_v1a.yml \
  --dataset_path data/ut-zap50k \
  2>&1 | tee output/utzappos_i2cd_light_v1a/train.log
```

### Results

Pending.

### Analysis

Pending.
