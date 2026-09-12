# 实验结果收集目录（`Code` 维护，原 Agent B）

> 按实验归类存放**汇总表 + 原始结果**；原始 JSON 在各自 `raw/` 子目录（gitignored，仅本地/服务器）。
> 汇总表（.md/.csv）为**最终进论文口径**（统一 GPU 基座 + 修复后代码）。供 `Paper` 直接引用。

## 目录索引

| 实验 | 状态 | 汇总文件 | 原始结果 |
|---|---|---|---|
| **E-1** 效率（并入 E-5） | ⬜ 待 E-5 | `E-1_efficiency/` | `E-1_efficiency/raw/` |
| **E-2** 消融（导师方案 4 组 × 5 数据集） | 🔄 CN 修复版重跑排队中（队列 #14/#15）；旧 20/20 作废 | `E-2_ablation/E2_ablation_summary.md`（旧数字作废） | `E-2_ablation/raw/` |
| **E-3** Patch 消融（P∈{1,3,5,7}） | ✅ CN 修复版重跑 8/8（2026-09-12） | `E-3_patch/E3_patch_summary.md` | `E-3_patch/raw/`（8 files） |
| **E-4** 时序（TE vs TD） | ✅ CN 修复版重跑（2026-09-12） | `E-4_time_decay/E4_time_decay_summary.md` | `E-4_time_decay/raw/` |
| **E-5** 显著性（双任务 × 5 种子） | ✅ CN 修复版重跑 10/10（2026-09-12）；汇总已更新 | `E-5_significance/E5_summary.md` | `E-5_significance/raw/`（10 files） |
| **主表**（CN 修复版重跑 + 基线） | 🔄 RT、RB 双任务完成（20 files）；WV sign✓/linksign 3/5；BA sign✓/linksign 2/5；OTC 排队 | `main_tables/` | `main_tables/raw/{sign,linksign}/{RedditHyperlinkTitle,RedditHyperlinkBody}` |

## 约定
- 所有表格数据 = **GPU 基座 + 修复后代码**（RAE bug 已修 + `.P` 标记命名 + JSON 含 `device`）。
- CPU 期数据仅存档/对照，不进本目录汇总。
- 原始 JSON 命名即配置指纹：`{Model}_seed{N}.NN-{n}.LF-{l}.RAS-{E/D}.RASE-{E/D}.BTE-{E/D}.CNAS-{E/D}.P{p}.{TE/TD}.json`
- 每份汇总表须含：数据集/任务/配置位/设备/seed，附一句结论。
- 归档同步：`python tools/sync/fetch_results.py --set e2|e3|e4|e5|main-a|all`（只读服务器；sha256 清单见 `_sync_raw_log.csv`）；关键指标快照：`python tools/verify/snapshot_results.py`。
