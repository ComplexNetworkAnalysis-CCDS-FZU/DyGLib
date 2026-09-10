# 实验结果收集目录（`Code` 维护，原 Agent B）

> 按实验归类存放**汇总表 + 原始结果**；原始 JSON 在各自 `raw/` 子目录（gitignored，仅本地/服务器）。
> 汇总表（.md/.csv）为**最终进论文口径**（统一 GPU 基座 + 修复后代码）。供 `Paper` 直接引用。

## 目录索引

| 实验 | 状态 | 汇总文件 | 原始结果 |
|---|---|---|---|
| **E-1** 效率（并入 E-5） | ⬜ 待 E-5 | `E-1_efficiency/` | `E-1_efficiency/raw/` |
| **E-2** 消融（导师方案 4 组 × 5 数据集） | 🔄 修复后重跑中（12/20） | `E-2_ablation/` | `E-2_ablation/raw/` |
| **E-3** Patch 消融（P∈{1,3,5,7}） | ✅ 完成（GPU） | `E-3_patch/E3_patch_summary.md` | `E-3_patch/raw/`（8 files） |
| **E-4** 时序（TE vs TD） | ✅ 完成（GPU） | `E-4_time_decay/` | `E-4_time_decay/raw/` |
| **E-5** 显著性（双任务 × 5 种子） | 🔄 修复后重跑中（队列自动派发） | `E-5_significance/` | `E-5_significance/raw/` |
| **主表**（修复后重跑 + 基线） | ⬜ 待执行 | `main_tables/` | `main_tables/raw/` |

## 约定
- 所有表格数据 = **GPU 基座 + 修复后代码**（RAE bug 已修 + `.P` 标记命名 + JSON 含 `device`）。
- CPU 期数据仅存档/对照，不进本目录汇总。
- 原始 JSON 命名即配置指纹：`{Model}_seed{N}.NN-{n}.LF-{l}.RAS-{E/D}.RASE-{E/D}.BTE-{E/D}.CNAS-{E/D}.P{p}.{TE/TD}.json`
- 每份汇总表须含：数据集/任务/配置位/设备/seed，附一句结论。
