# Agent 注册表（AGENTS REGISTRY）

> **2026-09-11 起启用「角色代号」**；字母 A/B/C 仅作历史别名。本表是 **Agent 身份与分工的唯一权威**（任务决策仍以 `docs/ADVISOR_DECISIONS.md` 为准）。
> 新 Agent 加入：**先在本表登记（代号 | 角色 | 工作区 | 职责边界），再开工**。

## 代号一览

| 代号 | 角色 | 工作区 | 职责边界 | 信箱分区 | 历史别名 |
|---|---|---|---|---|---|
| `Paper` | 论文正文/理论/回复信 | `D:\Sign_DygFormer` | 只改论文侧文件；不越界改代码/实验 | 给 Paper | A |
| `Code` | 代码/实验/数据/服务器 | `D:\codes\DyGLib`（本仓库） | 实验执行与交付、`docs/` 单一事实源维护、服务器/队列运维 | 给 Code | B |
| `Baseline` | DynamiSE/DySDGNN 基线复现 | `D:\codes\DynamiSE_DySDGNN_repro` | R2-5 方案① 复现；不改 DyGLib，数据只读 `processed_data/` | 给 Baseline | C |
| `Perf` | Rust + PyO3 内核加速（用户称「路线 C」） | 待创建（创建后更新本表） | **本地先行**：bit-exact 验证 + 提速实测达标后才上服务器；工作区独立 | 给 Perf | （2026-09-11 新加入） |

⚠️ 注意：「**路线 C**」（Rust 加速）与历史「Agent C / `Baseline`」**互不相干**，引用时请写清。

## 通信与事实源
- **信箱**：`D:\codes\DyGLib\docs\HANDOFF.md`（同机绝对路径，全体读写；署名用代号）。
- **进度/结果**：`docs/PROGRESS.md`（表格+结论，供直接引用）。
- **决策裁决**：`docs/ADVISOR_DECISIONS.md`（唯一权威）。
- 状态流转：⬜ 待办 → 🔄 进行 → ✅ 完成 / 🔴 重大（接收方处理后改）。

## 变更记录
- 2026-09-11：启用角色代号（`Paper`/`Code`/`Baseline`）；新 Agent `Perf`（Rust+PyO3 加速）登记。
