# E-4 时序编码消融汇总（TE vs TD，CN 真交集修复版，2026-09-12）

> 任务：link&sign（SignLinkPrediction，3 分类）；数据集：WikiVote@20000（NN-15, LF-10）；模型：SignDyGFormer（全模块 RAS+RAE+BTE+CNAS）；seed=42。
> 唯一变量 = 时间编码：**TE**（默认 Time Encoding） vs **TD**（Time Decay，`--time-decay-lambda 1.0`，队列 `@` 原生命令）。
> 对照口径：同一配置、同一 seed（TE 取 E-3 WV P1）。**CN 修复版重跑**；旧 TD 为 base 配置（RAS/RAE 关闭）不可比，已废弃。

## Test 指标表（seed=42）

| 时间编码 | AUC | AP | F1-wt | Sign-F1 |
|---|---|---|---|---|
| TE（默认） | 0.9607 | 0.8150 | 0.9032 | 0.8732 |
| TD（λ=1.0） | 0.9565 | 0.8030 | 0.9006 | 0.8614 |
| Δ (TD−TE) | -0.0042 | -0.0120 | -0.0026 | -0.0118 |

## 结论（供 Paper 引用）

1. **同配置下 TD 未优于 TE**：AUC / AP / Sign-F1 小幅落后（≤0.012），默认时间编码（TE）保留合理。
2. 单种子、单数据集（WikiVote）；结论强度以主表/显著性口径为准，本表用于时间编码选择的消融叙事。
3. 原始 JSON：`raw/SignDyGFormer_seed42.NN-15.LF-10.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TD.json`；TE 对照在 `results/E-3_patch/raw/` 同名 `P1.TE`。
