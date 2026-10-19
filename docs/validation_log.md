# CAMS-I2CD-Light Validation Log

This file records each validation version, the controlled change, and the observed result.

## Baseline: CAMS Original

Dataset: UT-Zappos, seed 0.

```text
closed-world:
S 68.62 | U 76.10 | HM 56.56 | AUC 43.76

open-world:
S 68.62 | U 64.83 | HM 52.13 | AUC 36.25
```

## v1: I2CD-light MVP

### Scheme

Add a visual-feature auxiliary head:

```text
composition feature -> disentangle attr/object
attr/object -> compose pair feature
pair feature -> re-disentangle attr/object
```

Training keeps CAMS inference unchanged and adds auxiliary CE, cycle, and anchor losses.

### Weights

```text
lambda_i2cd_dis    = 0.10
lambda_i2cd_comp   = 0.10
lambda_i2cd_re     = 0.05
lambda_i2cd_cycle  = 0.05
lambda_i2cd_anchor = 0.05
```

### Result

UT-Zappos seed 0, closed-world test with `val_best.pt`:

```text
S 69.01 | U 73.88 | HM 54.79 | AUC 42.93
attr_acc 51.85 | obj_acc 73.30
```

UT-Zappos seed 0, open-world test with `final_model.pt`:

```text
S 69.01 | U 63.67 | HM 50.83 | AUC 36.60
attr_acc 51.72 | obj_acc 73.20
```

### Analysis

The auxiliary loss is active, but closed-world AUC drops from 43.76 to 42.93. Open-world AUC has only a small gain, while unseen/HM drop. This version is not a stable improvement.

## v1a: Lower I2CD Loss Weights

### Scheme

Keep the v1 architecture unchanged and reduce all I2CD auxiliary weights by 5x.

### Weights

```text
lambda_i2cd_dis    = 0.02
lambda_i2cd_comp   = 0.02
lambda_i2cd_re     = 0.01
lambda_i2cd_cycle  = 0.01
lambda_i2cd_anchor = 0.01
```

### Result

UT-Zappos seed 0, closed-world test with `val_best.pt`:

```text
S 68.62 | U 73.82 | HM 54.78 | AUC 42.65
attr_acc 51.37 | obj_acc 73.13
```

Best validation epoch: epoch 4.

```text
S 70.81 | U 72.66 | HM 63.75 | AUC 48.47
```

### Analysis

Lowering the auxiliary weights does not recover performance. The result is slightly worse than v1, so the main issue is probably not simply over-regularization strength.

## v1b: Remove Anchor Loss

### Scheme

Keep v1 dis/comp/re/cycle losses, but remove `anchor_loss`. This tests whether forcing the learned I2CD factors to stay close to CAMS attribute/object image branches limits useful disentanglement.

### Weights

```text
lambda_i2cd_dis    = 0.10
lambda_i2cd_comp   = 0.10
lambda_i2cd_re     = 0.05
lambda_i2cd_cycle  = 0.05
lambda_i2cd_anchor = 0.00
```

### Command

```bash
cd ./lyt/czsl/CAMS-I2CD-Light
source ./anaconda3/etc/profile.d/conda.sh
conda activate crdiff
mkdir -p output/utzappos_i2cd_light_v1b

CUDA_VISIBLE_DEVICES=3 python -u train.py \
  --config configs/utzappos_i2cd_light_v1b.yml \
  --dataset_path data/ut-zap50k \
  2>&1 | tee output/utzappos_i2cd_light_v1b/train.log
```

### Result

UT-Zappos seed 0, closed-world test with `val_best.pt`:

```text
S 69.11 | U 73.82 | HM 54.39 | AUC 42.85
attr_acc 51.92 | obj_acc 73.06
```

Best validation epoch: epoch 4.

```text
S 71.04 | U 73.09 | HM 64.18 | AUC 48.92
```

### Analysis

Removing the anchor loss does not improve test performance. It gives the best validation AUC among v1/v1a/v1b, but the final closed-world test AUC is still below the CAMS baseline and slightly below v1. This suggests the anchor term is not the main cause of the performance drop. The visual-feature I2CD auxiliary head still fails to transfer validation gains to test pairs.

## v2: Prompt-Space I2CD

### Scheme

Stop validating the visual-feature I2CD head and move the disentangle-compose-redisentangle idea into prompt space.

For each pair prompt:

```text
composition prompt -> disentangle attr/object prompts
attr prompt + object prompt -> compose a pair prompt
composed pair prompt -> re-disentangle attr/object prompts
```

The composed prompt is fused with the original CAMS composition prompt and is used in both training and inference. This is different from v1/v1a/v1b, where the I2CD head only acted as an auxiliary visual-feature regularizer.

### Weights

```text
lambda_prompt_i2cd_compose = 0.050
lambda_prompt_i2cd_dis     = 0.050
lambda_prompt_i2cd_re      = 0.025
lambda_prompt_i2cd_cycle   = 0.025
prompt_i2cd_mix_init       = 0.200
```

### Command

```bash
cd ./lyt/czsl/CAMS-I2CD-Light
source ./anaconda3/etc/profile.d/conda.sh
conda activate crdiff
mkdir -p output/utzappos_prompt_i2cd_v2

CUDA_VISIBLE_DEVICES=3 python -u train.py \
  --config configs/utzappos_prompt_i2cd_v2.yml \
  --dataset_path data/ut-zap50k \
  2>&1 | tee output/utzappos_prompt_i2cd_v2/train.log
```

### Result

UT-Zappos seed 0, closed-world test with `val_best.pt`:

```text
S 67.64 | U 74.19 | HM 54.43 | AUC 42.25
attr_acc 51.82 | obj_acc 72.20
```

Best validation epoch: epoch 4.

```text
S 70.35 | U 73.64 | HM 64.37 | AUC 48.68
```

The learned prompt fusion ratio stayed close to the initialization:

```text
p_mix: 0.200 -> about 0.197
```

### Analysis

The first prompt-space implementation does not improve UT-Zappos. It performs worse than the CAMS baseline and also worse than visual-feature v1/v1b on closed-world AUC.

The likely issue is not that prompt-space I2CD is completely inactive. The log shows `p_i2cd`, `p_comp`, and `p_mix`, and the refined prompt is used in training and inference. The issue is more likely that the composed prompt is mixed into CAMS from epoch 1 with a relatively large 20% ratio while the composer is still untrained. This can disturb the original CAMS prompt alignment before the prompt-I2CD module has learned a useful composition mapping.

The next prompt-space ablation should reduce or delay this disturbance rather than abandon prompt-space immediately:

```text
v2a: smaller mix_init, e.g. 0.05, and lower prompt-I2CD weights
v2b: warm up prompt-I2CD as an auxiliary loss first, then enable prompt fusion
v2c: keep fusion disabled in training and only evaluate refined prompts after the module is learned
```

## v2a: Conservative Prompt Fusion

### Scheme

Keep the v2 prompt-space I2CD architecture unchanged, but reduce how much the composed prompt can disturb CAMS at the start of training.

Compared with v2:

```text
prompt_i2cd_mix_init:       0.20  -> 0.05
lambda_prompt_i2cd_compose: 0.050 -> 0.020
lambda_prompt_i2cd_dis:     0.050 -> 0.020
lambda_prompt_i2cd_re:      0.025 -> 0.010
lambda_prompt_i2cd_cycle:   0.025 -> 0.010
```

### Command

```bash
cd ./lyt/czsl/CAMS-I2CD-Light
source ./anaconda3/etc/profile.d/conda.sh
conda activate crdiff
mkdir -p output/utzappos_prompt_i2cd_v2a

CUDA_VISIBLE_DEVICES=3 python -u train.py \
  --config configs/utzappos_prompt_i2cd_v2a.yml \
  --dataset_path data/ut-zap50k \
  2>&1 | tee output/utzappos_prompt_i2cd_v2a/train.log
```

### Result

Pending.

### Analysis

Pending.
