# DirG 首单 · 毕业论文「第二点（有向图）」开工任务包

> 建立：2026-09-22（用户拍板四项边界后）。维护：`Code`。权威层级：凡涉及方向/约束/优先序，以 `docs/ADVISOR_DECISIONS.md` 与用户逐次指示为准；本文件 = 执行层任务包。

## 0. 已确认边界（2026-09-22 用户拍板）

| 项 | 决定 |
|---|---|
| 研究范围 | **A：有向符号图方法** —— 复用 `DirectSignDyGFormer`（方向感知 in/out 序列 + Status Theory 有向 2-路径，不确定时退避 Balance Theory）与 `DirectedNeighborSampler`；目标：方向感知序列 + 有向符号预测 |
| 数据集 | **BA、OTC**（天然有向）+ **RT、RB、WV**（复用主线口径）；`myket` 已预处理（有向）备用 |
| 算力/服务器 | **允许直连服务器（仍须用户逐次明确许可）**；**当前修订 ours 批次全部完成后匀出 1 张卡**（预计 09-23 晚～09-24，以 `Code` 确认窗口为准） |
| 时间线 | **2027-04 前送审**，需预留修改返修时间 → 内部里程碑见 §4 |

## 1. 任务包（T0–T5）

- **T0 脚手架（0.5 天）**：读权威文档（本文件、DyGLib `AGENTS.md`、`docs/AGENTS_REGISTRY.md`、`docs/ADVISOR_DECISIONS.md`、`docs/HANDOFF.md`）；确认信箱 MCP 可用（`.vscode/mcp.json`，`MAILBOX_AGENT=DirG`）；在自己仓建 `REPORTS/` 进度档。
- **T1 协议冻结（1 天）**：自己仓 `docs/DIR_TASK_SPEC.md`：
  - 任务定义：有向存在性（u→v）与有向符号（正/负）；反边（rev）与随机负（neg）协议；
  - 数据集与划分：BA/OTC/RT/RB/WV（复用主线 `val_ratio/test_ratio/tail_num` 口径）+ myket 备用；
  - 指标：`f1_binary`/`auc`/`ap`/方向判定准确率（若有）等，主指标待与用户/导师确认后冻结；
  - 判据：Δ≥0.005（或与用户约定阈值）+ 同种子配对显著性 + ≥3/5 同向；
  - 代际标记规范：结果名后缀（如 `.DR-*`），默认关 = 零行为变更。
- **T2 资产审计（1–2 天）**：走读 `train_direct_link_prediction.py`、`models/DirectSignDyGFormer.py`、`models/DirectStatusEncoder.py`、`utils/direct_neighbor_sampler.py`；
  - **有向护栏单测**（参照 `tools/verify/test_sampling_guard.py` 风格）：严格 past（`t < t_query`）、目标边自身排除、in/out 序列分离、反边不泄露；
  - smoke：本地 CPU 小样本冻结验证（≤1k 边）；正式 smoke 走服务器（窗口内）。
- **T3 基线矩阵（服务器，窗口内）**：{DyGFormer, TGAT, GraphMixer, SignDyGFormer, DirectSignDyGFormer} × {有向采样, 无向采样} × {BA, OTC, RT, RB, WV}；**先 1 种子初筛 → 判据过 → 5 种子**。
- **T4 方法改进（1–2 个，单变量 + 代际标记）**：候选——
  (a) 有向 BTE/Status 门控（主线 BTE 门控接口预留备注即"后续或有向图用"）；
  (b) 有向 CNAS（in-in/out-out/in-out 三类 2-路径配额）；
  (c) 方向不对称时间衰减。
- **T5 交付**：结果表 + 结论（数据集 × 指标 mean±pstd + 同种子配对 Δ/t/p）→ 信箱 → `Code` 登记 DyGLib `docs/PROGRESS.md` → 供毕业论文第二章。

## 2. 与 `Code` 的接口

- **正式实验**：DirG 出「命令清单（含 `@GPU@` 形态）+ 预期产物名 + 判据」→ `Code` 插入队列/部署并回报；若 DirG 自跑，须**用户当次许可**且遵守队列纪律（不抢卡、不覆盖他人产物）。
- **取数**：复用 DyGLib `tools/sync/fetch_results.py`（只读）；或 `Code` 代取。
- **结果登记**：DyGLib `docs/PROGRESS.md` 由 `Code` 登记（单一事实源）；DirG 自己仓 `REPORTS/` 维护过程档。

## 3. 资源与护栏

- 服务器：逐次许可；GPU 窗口由 `Code` 协调（**当前修订批次优先**）。
- 数据：`processed_data/`、`DG_data/` 为 junction 至 DyGLib（**只读**，勿写入）；新预处理数据（如 Epinions）放自己仓 `preprocess_data/`，服务器同步一律走 git（禁 scp）。
- 红线：不碰 `sign-adoption` 分支与既有主表产物；不改 DyGLib 工作区文件（经信箱请求对方执行）。

## 4. 里程碑（内部，倒排 2027-04 送审）

| 时间 | 目标 |
|---|---|
| 09-24 ~ 09-30 | T0–T2 完成（协议冻结 + 审计 + 护栏 + smoke）；接卡 |
| 10 月底 | 基线矩阵 5 种子齐 + 方法改进方案定稿（1–2 个） |
| 12 月底 | 方法实验（初筛→5 种子）齐，含消融/鲁棒性 |
| 01 月底 | 论文数据定稿（表格/图） |
| 02–03 月 | 写作/插图 + 预留返修缓冲 |

## 5. 首次回报要求（T0–T2 完成时）

信箱发 `Code`：协议冻结稿摘要 + 审计发现（bug/差异清单）+ 护栏单测结果 + smoke 数字 + T3 排程请求（含估时）。
