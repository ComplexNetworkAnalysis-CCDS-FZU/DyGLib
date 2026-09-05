# SignDyG 修订项目 — 文档索引

> 项目：SignDyG（NEUCOM-D-26-13975），修订截止 2026-10-10。
> 分工：**Agent A** = 论文正文/理论/回复信；**Agent B** = 代码/实验/数据。
> 命令约定：本目录内文档中的命令默认在**仓库根目录**执行。

## 文档角色与权威层级

| 文档 | 角色 | 权威性 | 更新频率 |
|---|---|---|---|
| **`ADVISOR_DECISIONS.md`** | ★ 与导师讨论后的**决策记录** | **唯一权威**（与此冲突时以此为准） | 导师讨论后更新 |
| `REVIEWER_COMMENTS.md` | 两位审稿人**完整意见归档** + 应对映射 | 参考（原始意见） | 一次性归档 |
| `EXPERIMENT_PLAN.md` | **实验运行计划**（顺序、命令、数据集） | 执行依据（随决策更新） | 计划变化时 |
| `PROGRESS.md` | **进度状态**（各实验状态、结果、更新日志） | 状态记录 | 每次实验后 |
| `REPORT_TO_ADVISOR.md` | 导师汇报（2026-09-01 版） | 历史快照 | 低频 |

## 决策流向（更新顺序）

```
导师讨论
   ↓ 记录
ADVISOR_DECISIONS.md  （唯一权威）
   ↓ 据此修订
EXPERIMENT_PLAN.md → 执行实验
                       ↓ 完成后
                    PROGRESS.md（状态+结果）
                       ↓ 汇总
                    交付 Agent A
```

## 常用路径备忘

| 内容 | 路径 |
|---|---|
| 实验批量入口 | `run_experiments.py`（根目录） |
| 统计显著性脚本 | `dataset_analysis/compute_stats.py` |
| 噪声模块（可迁移 SEMBA） | `utils/noise.py` |
| 服务器数据就绪脚本 | `server_setup.sh`（根目录） |
| 结果 JSON | `saved_results/{LinkSign\|SignLinkPrediction}/{model}/{dataset}/` |
| 批量运行日志 | `expm-YYYY-MM-DD-logs/{任务}/` |
