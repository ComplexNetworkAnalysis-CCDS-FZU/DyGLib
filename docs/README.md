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
| `HANDOFF.md` + `D:\codes\agent-mailbox` | **统一信箱**（2026-09-13 起：独立仓 `agent-mailbox`，JSONL 真源 + CLI/MCP 工具；`HANDOFF.md` 为指针页；旧 `docs/handoff/*` 只读归档） | 消息记录（非权威） | 任务结束/有新请求时 |
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

## 多 Agent 协作（2026-09-05 起，Agent C 2026-09-09 加入）

> 仓库根 `AGENTS.md` = 协作契约（本工作区会话自动加载）；
> Agent A 工作区 `D:\Sign_DygFormer` 与 Agent C 工作区 `D:\codes\DynamiSE_DySDGNN_repro` 根各有 `copilot-instructions.md` 指针（绝对路径指向本目录）。
> 信息流：`agent → 统一信箱（D:\codes\agent-mailbox，只经 CLI/MCP 工具写入）/ PROGRESS.md → agent`（2026-09-13 切换；规则见 `HANDOFF.md` 指针页与新仓 `README.md`），无需用户逐条转述。
> 单一事实源 = 本目录；同机绝对路径读写即时互见。
> **Agent C**：DynamiSE/DySDGNN 基线复现（R2-5 方案①），权威 = 其工作区 `IMPLEMENTATION_SPEC.md`，数据只读本仓库 `processed_data/`。

## 常用路径备忘

| 内容 | 路径 |
|---|---|
| 实验批量入口 | `run_experiments.py`（根目录） |
| 统计显著性脚本 | `dataset_analysis/compute_stats.py` |
| 噪声模块（可迁移 SEMBA） | `utils/noise.py` |
| 服务器数据就绪脚本 | `server_setup.sh`（根目录） |
| 结果 JSON | `saved_results/{LinkSign\|SignLinkPrediction}/{model}/{dataset}/` |
| 批量运行日志 | `expm-YYYY-MM-DD-logs/{任务}/` |
