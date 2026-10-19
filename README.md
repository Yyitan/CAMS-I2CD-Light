# CAMS-I2CD-Light：I2CD-Light（disentangle–compose–redisentangle 一致性辅助约束的 CAMS 改进）

> 基于 [CAMS](https://github.com/ybyangjing/CAMS)（[arXiv:2511.16378](https://arxiv.org/abs/2511.16378)）的改进研究 · 个人研究项目

## 研究动机

CAMS 的组合分支特征由交叉熵直接监督，缺少显式的"属性/对象可分解"结构约束。受 I2CD 思想启发，本仓库在保持 CAMS 推理路径不变的前提下，加入一个 **disentangle–compose–redisentangle** 循环一致性约束：把组合特征拆成属性/对象特征，重新组合成 pair 特征，再对重组特征做二次拆分，用 CE、cycle、anchor 等辅助损失约束整条链路，希望以此正则化 CAMS 特征、提升未见组合泛化。后续又将同一思想迁移到 prompt space（v2）。**需要如实说明：截至当前记录，视觉特征版与 prompt 版均未带来稳定提升，本仓库更多是一组带对照的负结果实验记录。**

## 方法

| 版本 | 模块 | 思路 |
|------|------|------|
| v1 / v1a / v1b | `model/i2cd_light.py` 的 `I2CDLight`（视觉特征辅助头） | `com 特征 → 拆分 attr/obj → compose 重组 pair 特征 → 再拆分`；辅助损失 `lambda_i2cd_dis / comp / re / cycle / anchor`，CAMS 推理不变。v1a 将全部权重降为 1/5；v1b 去掉 anchor 项 |
| v2 / v2a | `model/prompt_i2cd.py` 的 `PromptI2CD`（prompt space 版） | 对每对 pair prompt 做 disentangle–compose–redisentangle，将重组 prompt 以可学习比例 `prompt_i2cd_mix`（sigmoid 门控，初始 0.2）融合进 CAMS 组合 prompt，训练与推理都使用 |

辅助损失（v1 默认权重）：`lambda_i2cd_dis=0.10`、`lambda_i2cd_comp=0.10`、`lambda_i2cd_re=0.05`、`lambda_i2cd_cycle=0.05`、`lambda_i2cd_anchor=0.05`；v2 的 prompt 版为 `compose=0.05、dis=0.05、re=0.025、cycle=0.025、mix_init=0.2`。

## 仓库内容

本仓库**只包含新增/修改的部分，不含 CAMS 上游原始代码**。对上游文件（`train.py`、`test.py`、`model/cams.py`、`model/configure_model.py`、`configs/utzappos.yml`）的修改以 diff 形式记录在 `patches_from_upstream/` 中。

```
CAMS-I2CD-Light/
├── model/
│   ├── i2cd_light.py               # I2CDLight 视觉特征辅助头（核心新增）
│   └── prompt_i2cd.py              # PromptI2CD prompt-space 模块（核心新增）
├── configs/
│   ├── mit_i2cd_light.yml          # MIT-States v1 配置
│   ├── utzappos_i2cd_light.yml     # UT-Zappos v1
│   ├── utzappos_i2cd_light_v1a.yml # v1a：权重×1/5
│   ├── utzappos_i2cd_light_v1b.yml # v1b：去 anchor
│   ├── utzappos_prompt_i2cd_v2.yml # v2：prompt space
│   └── utzappos_prompt_i2cd_v2a.yml# v2a：保守融合（mix_init=0.05，待实验）
├── docs/
│   ├── I2CD_Light_experiment_report.md  # 实验报告（动机/改动/命令/结果/分析）
│   └── validation_log.md                # 各版本验证记录与结论
└── patches_from_upstream/          # 对上游文件的修改记录（git apply 用）
    ├── train.py.diff               # 自动创建 save_path、I2CD loss 进度条诊断
    ├── test.py.diff
    ├── model__cams.py.diff         # 挂载 I2CDLight / PromptI2CD 辅助损失
    ├── model__configure_model.py.diff
    └── configs__utzappos.yml.diff
```

## 实验结果

以下数字均来自 `docs/validation_log.md` / `docs/I2CD_Light_experiment_report.md`（UT-Zappos，seed 0）。

**CAMS 基线**：

| 设置 | S | U | HM | AUC |
|------|------|------|------|------|
| closed-world（`val_best.pt`） | 68.62 | 76.10 | 56.56 | 43.76 |
| open-world（`final_model.pt`） | 68.62 | 64.83 | 52.13 | 36.25 |

**各版本（closed-world，`val_best.pt`）**：

| 版本 | S | U | HM | AUC | 结论 |
|------|------|------|------|------|------|
| v1（视觉特征辅助头） | 69.01 | 73.88 | 54.79 | 42.93 | AUC 较基线 -0.83，U/HM 下降；open-world AUC 36.60（+0.35），非稳定提升 |
| v1a（权重×1/5） | 68.62 | 73.82 | 54.78 | 42.65 | 略差于 v1，说明问题不单纯是正则过强 |
| v1b（去 anchor） | 69.11 | 73.82 | 54.39 | 42.85 | 验证集 AUC 最高（48.92）但测试仍低于基线，anchor 非主因 |
| v2（prompt space） | 67.64 | 74.19 | 54.43 | 42.25 | 差于基线与 v1/v1b；`p_mix` 基本停在初始值 0.20 附近（0.197） |

**负结果分析**（详见 validation_log）：辅助损失确实在下降、模块在训练，但视觉特征版的验证集收益无法迁移到测试 pair；prompt 版的初步推断是未经训练的 composer 从第 1 个 epoch 就以 20% 比例混入 prompt，扰动了 CAMS 原有 prompt 对齐。`v2a`（mix_init=0.05 + 更低权重）配置已就绪，结果待补。

## 复现步骤

```bash
# 1. 克隆上游 CAMS（本仓库不包含上游代码）
git clone https://github.com/ybyangjing/CAMS.git CAMS-I2CD-Light-work
cd CAMS-I2CD-Light-work

# 2. 应用对上游文件的修改
git apply /path/to/CAMS-I2CD-Light/patches_from_upstream/*.diff

# 3. 放入本仓库新增文件
#    model/i2cd_light.py、model/prompt_i2cd.py、configs/、docs/
#    按相同相对路径复制进 CAMS-I2CD-Light-work/

# 4. 准备数据集并修改 config 中 clip_model 路径（默认 ./lyt/czsl/CLIP-main/ViT-L-14.pt）
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --config configs/utzappos_i2cd_light.yml \
  --dataset_path /path/to/ut-zap50k \
  2>&1 | tee output/utzappos_i2cd_light/train.log

# 5. 测试（closed-world 用 val_best.pt；open-world 加 --open_world，配 final_model.pt）
CUDA_VISIBLE_DEVICES=0 python test.py \
  --config configs/utzappos_i2cd_light.yml \
  --dataset_path /path/to/ut-zap50k
```

训练日志中的 `i2cd / dis / comp` postfix 即辅助损失诊断项。环境依赖与数据集格式与上游 CAMS 一致。

## 引用

本仓库基于以下工作，如使用请同时引用：

```bibtex
@article{yang2025cams,
  title={CAMS: Towards Compositional Zero-Shot Learning via Gated Cross-Attention and Multi-Space Disentanglement},
  author={Yang, Pan and Deng, Cheng and Yang, Jing and Zhao, Han and Liu, Yun and Chen, Yuling and Ruan, Xiaoli and Chen, Yanping},
  journal={arXiv preprint arXiv:2511.16378},
  year={2025}
}
```

## License

本项目以 MIT License 发布（仅覆盖本仓库新增的文件；上游 CAMS 代码版权归原作者所有，且上游仓库未声明 LICENSE）。
