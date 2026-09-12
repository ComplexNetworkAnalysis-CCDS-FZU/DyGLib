# 📬 交接信箱（HANDOFF）· 入口页

> **2026-09-12 拆箱**：信箱由「单文件多写者」改为「**一人一箱**」——旧模式已被旧缓冲**互相覆盖 2 次**（13:52 / 22:02–22:05，均自 git 恢复）。
> ⛔ **本文件仅 `Code` 维护**（导航 + 速览）；**任何 Agent 请勿往本文件写条目**——写到你**自己的发件箱**（见下表）。
> 正式结果/数据仍走 `docs/PROGRESS.md`；历史条目（≤2026-09-10）在 `docs/HANDOFF_ARCHIVE.md`。

## 导航：谁写哪、读哪

| 角色 | **你的发件箱（唯一写点）** | 你的收件 = 其余三箱中「收件人=你」的行 |
|---|---|---|
| `Code`（B） | `docs/handoff/outbox-code.md` | `outbox-paper.md` · `outbox-baseline.md` · `outbox-perf.md` |
| `Paper`（A） | `docs/handoff/outbox-paper.md` | `outbox-code.md` · `outbox-baseline.md` · `outbox-perf.md` |
| `Baseline`（C） | `docs/handoff/outbox-baseline.md` | `outbox-code.md` · `outbox-paper.md` · `outbox-perf.md` |
| `Perf` | `docs/handoff/outbox-perf.md` | `outbox-code.md` · `outbox-paper.md` · `outbox-baseline.md` |

## 协议（三步）

1. **发**：任务结束/有新请求 → 往**自己的**发件箱追加一行：`| 日期 | 收件人 | 状态 | 内容 |`（状态 ⬜待办 / 🔄进行 / ✅完成 / 🔴重大）。
2. **收**：处理别人发给你的条目后 → 在**自己的**发件箱加一条**回执行**（收件人=原发件人、✅、一句话结论）——**切勿编辑对方的文件**。
3. **结**：原发件人看到回执 → 把**自己**那条改为 ✅。

> 每个箱只有唯一写入者 = 不会再互相覆盖；本页速览由 `Code` 周期刷新，一切以各箱原文为准。

## 🔴 活跃事项速览（2026-09-12 拆箱快照）

- **→ Paper ⬜**：k×N 热力图（10 张）是否修复版重跑（问询）；E-3/E-4/E-5 修复版交付已发
- **→ Code ⬜**：Paper 补充实验请求（①BTE 消融 ②sign 重调参 ③旧码对照 ④BA/WV sign 同步）——待用户批准后编排执行
- **→ Baseline ⬜**：①接入汇报 ②第二波方法评估（R1-4）③服务器硬约束
- **→ Perf ⬜**：①真实数据提速复核 ②M4 集成设计（用户优先推进 Rust）

## 全局备注

- 工作区分布：`Paper`=`D:\Sign_DygFormer` · `Code`=本仓库 · `Baseline`=`D:\codes\DynamiSE_DySDGNN_repro` · `Perf`=`D:\codes\SignDyG-Perf`（同机绝对路径互见；改动请在 DyGLib 仓库内提交，或由 `Code` 代提交）。
- **服务器纪律（2026-09-11）**：服务器访问**唯一通道 = `Code`**，且须用户**逐次明确许可**；`Paper`/`Baseline`/`Perf` 一律禁止接触服务器（含自动运行 agent）。
- **本地网络代理（2026-09-12，用户提供）**：`http://192.168.10.3:7897`（Clash）——供各 agent 访问 GitHub 等外部资源（Code 实测 GitHub API 200 ✓）；仅网络访问，**不改变服务器纪律**。
- 代号 ↔ 历史别名：`Paper`=A · `Code`=B · `Baseline`=C · `Perf`（无旧名）；详见 `docs/AGENTS_REGISTRY.md`。
- 状态由**发件人**维护自己的行（协议第 3 步）；收件方以「回执行」反馈处理结果。
