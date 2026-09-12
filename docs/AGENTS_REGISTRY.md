# Agent 注册表（AGENTS REGISTRY）

> **2026-09-11 起启用「角色代号」**；字母 A/B/C 仅作历史别名。本表是 **Agent 身份与分工的唯一权威**（任务决策仍以 `docs/ADVISOR_DECISIONS.md` 为准）。
> 新 Agent 加入：**先在本表登记（代号 | 角色 | 工作区 | 职责边界），再开工**。

## 代号一览

| 代号 | 角色 | 工作区 | 职责边界 | 发件箱（唯一写点） | 历史别名 |
|---|---|---|---|---|---|
| `Paper` | 论文正文/理论/回复信 | `D:\Sign_DygFormer` | 只改论文侧文件；不越界改代码/实验 | `outbox-paper.md` | A |
| `Code` | 代码/实验/数据/服务器 | `D:\codes\DyGLib`（本仓库） | 实验执行与交付、`docs/` 单一事实源维护、服务器/队列运维 | `outbox-code.md` | B |
| `Baseline` | DynamiSE/DySDGNN 基线复现 | `D:\codes\DynamiSE_DySDGNN_repro` | R2-5 方案① 复现；不改 DyGLib，数据只读 `processed_data/`；⛔ **禁服务器，只能本地**（服务器侧动作走 `Code`+用户许可） | `outbox-baseline.md` | C |
| `Perf` | Rust + PyO3 内核加速（用户称「路线 C」） | `D:\codes\SignDyG-Perf`（2026-09-11 创建） | ⛔ **不得接触服务器，只能本地运行**（未经用户逐次明确许可；含自动运行 agent；闸门达标≠许可）；DyGLib **只读** | `outbox-perf.md` | （2026-09-11 新加入） |

⚠️ 注意：「**路线 C**」（Rust 加速）与历史「Agent C / `Baseline`」**互不相干**，引用时请写清。

## 通信与事实源
- **信箱（2026-09-12 拆箱：一人一箱）**：入口/速览 = `D:\codes\DyGLib\docs\HANDOFF.md`（`Code` 维护，他方只读）；**各 Agent 的发件箱 = `D:\codes\DyGLib\docs\handoff\outbox-{paper|code|baseline|perf}.md`**（唯一写入者=箱主，署名用代号；收件方在自己箱内回执）。
- **进度/结果**：`docs/PROGRESS.md`（表格+结论，供直接引用）。
- **决策裁决**：`docs/ADVISOR_DECISIONS.md`（唯一权威）。
- **服务器访问**：唯一通道 = `Code`，且须**用户逐次明确许可**；`Paper`/`Baseline`/`Perf` 一律禁止接触服务器（含自动运行的 Copilot/后台 agent）。
- 状态流转：⬜ 待办 → 🔄 进行 → ✅ 完成 / 🔴 重大（发件人维护自己行；收件方处理后在自箱回执）。

## 变更记录
- 2026-09-12：**信箱拆分为「一人一箱」**（`docs/handoff/outbox-{paper,code,baseline,perf}.md`；入口页=`docs/HANDOFF.md`），消除单文件多写者覆盖（当日曾 2 次事故）；各 Agent 工作区契约指针同步更新。
- 2026-09-11：启用角色代号（`Paper`/`Code`/`Baseline`）；新 Agent `Perf`（Rust+PyO3 加速）登记。
- 2026-09-11：`Perf` 工作区创建于 `D:\codes\SignDyG-Perf`（ref-only 交付：参考实现 + golden fixtures + 上游逐位对照；crate 由 `Perf` 自建）。
- 2026-09-11：用户硬约束——`Perf` **严格禁止接触服务器，只能本地运行**（含自动运行 agent；未经用户逐次明确许可）。已固化至其契约/任务清单/安装文档。
- 2026-09-11：硬约束扩展——`Baseline` 同 `Perf` **严格禁止接触服务器**（只能本地）；明确原则：**服务器访问唯一通道 = `Code`（须用户逐次明确许可）**，其他 agent（含自动运行）一律禁止。
