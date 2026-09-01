# SignDyG 修订实验进度记录（NEUCOM-D-26-13975）

> **维护者**：Agent B（代码与实验）
> **用途**：所有 Agent（含 Agent A 论文撰写）通过本文件了解最新进度。
> **规则**：每次实验状态变化立即更新本文件，并通过 git 同步（本地 push → 服务器 pull）。
> 提交截止：2026-10-10。

## 全局状态（2026-09-01 更新）

| 项目 | 状态 | 说明 |
|---|---|---|
| 服务器 | ✅ 可用 | 2026-09-01 恢复访问 |
| 代码同步 | ✅ 完成 | 已推送 `sign-adoption` 分支至服务器裸仓库并 clone |
| 数据就绪 | ⬜ 待执行 | 运行 `bash server_setup.sh ~/旧副本路径` |
| 环境安装 | ⬜ 待执行 | `conda env create -f environment.yaml`（torch 按 CUDA 装） |
| E-7 噪声 | ⛔ 本轮不做 | 模块 `utils/noise.py` 已实现保留 |

## 实验状态总览

| 实验 | 任务 | 数据集 | 状态 | 结果路径 / 备注 |
|---|---|---|---|---|
| E-1 效率 | sign | RedditTitle@20000 | ⬜ 待执行 | 并入 E-5 seed42，不单独跑 |
| E-2 消融 | linksign | RedditTitle@20000 + WikiVote@20000 | ⬜ 待执行 | 5 组增量，10 runs |
| E-3 Patch | linksign | WikiVote@20000 + RedditBody@20000 | ⬜ 待执行 | P∈{1,3,5,7}，8 runs |
| E-4 时序 | linksign | WikiVote@20000 | ⬜ 待执行 | TE vs TD，2 runs |
| E-5 显著性 | sign + linksign | RedditTitle@20000 | ⬜ 待执行 | 各 5 种子，10 runs |
| 主表重跑 | sign + linksign | 5 数据集 | ⬜ 待执行 | 先 BitcoinAlpha 影响评估 |
| E-6 异配图 | — | — | ⛔ 本轮不做 | — |

**执行顺序（固定，不跳步）**：E-1(并入E-5) → E-2 → E-3 → E-4 → E-5 → 主表重跑

## 每次运行后需记录

对每个 run（或每个实验），记录：
- **开始/结束时间**、耗时
- **异常情况**（如有）
- **结果文件路径**（`saved_results/...`）与关键指标
- 更新上方"实验状态总览"表格

## 已确认决策（2026-08-20）

- 数据集统一 **tail 20000**（RedditTitle / RedditBody / WikiVote；BitcoinAlpha/BitcoinOTC 用全量）
- E-2/E-3/E-4 用 **link&sign** 任务；E-5 用 **sign + link&sign 双任务 × 5 种子**（满足审稿人 R2#8）
- **RAE bug 已修复**（`np.append` 未赋值）：旧主表（sign-ms.csv/linksign_ms.csv）用旧代码跑出，须用修复后代码重跑；先 BitcoinAlpha 影响评估（阈值 0.5%）再决定全量
- E-6（异配图）、E-7（噪声）：本轮不做
- SEMBA 在独立仓库，不在此实现

## 运行日志与结果位置

| 内容 | 路径 |
|---|---|
| run_experiments 批量日志 | `expm-YYYY-MM-DD-logs/{任务}/` |
| 结果 JSON（含 E-1 效率 4 项） | `saved_results/{LinkSign\|SignLinkPrediction}/{model}/{dataset}/` |
| profiler 推理明细 | 同目录 `{...}-profiler.json` |
| 统计汇总（mean±std + p 值） | `dataset_analysis/compute_stats.py` 输出 |

## 最近更新记录

- **2026-09-01**：服务器恢复；代码推送至服务器；`server_setup.sh`（数据就绪脚本）+ 本进度文件建立；E-7 确定本轮不执行。
- **2026-08-20**：E-1~E-7 全部代码实现完成（时间衰减、效率测量、消融/patch 脚本、统计脚本、噪声模块、RAE bug 修复），本地 CPU 冒烟测试全部通过；`EXPERIMENT_PLAN.md` 运行计划定稿。
