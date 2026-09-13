# 📬 交接信箱（HANDOFF）· 指针页（2026-09-13 起）

> ⚠️ **信箱已迁移至独立仓**——正式通道 = `D:\codes\agent-mailbox`（本页仅存指针；旧 `docs/handoff/*` 已于 2026-09-13 冻结为只读归档）。

## 新通道（唯一正式）

| 事项 | 位置 / 命令 |
|---|---|
| 真源（JSONL，append-only） | `D:\codes\agent-mailbox\data\outbox-<slug>.jsonl` |
| 渲染视图（人读） | `D:\codes\agent-mailbox\handoff\outbox-<slug>.md` |
| **读**（收件） | `python -m mailbox.cli list --for <自己> --open`（或看渲染视图） |
| **写**（只经工具，禁手编） | `send --from <角色> --to <对方> --type … --text …`；回执 `ack --by <角色> --ref <ID> --note …`；状态由**原发件人** `set-status` |
| 设计/命令速览 | `D:\codes\agent-mailbox\README.md` ｜ 实施记录 `TASK_LIST.md` |
| 箱名 = 代号 | `Code` / `Paper` / `Baseline` / `Perf` |

> 三步协议不变（发 → 回执 → 结）；速览改为新箱 `list --open`。
> ⛔ **旧 `docs/handoff/outbox-*.md`（v1，2026-09-12~09-13）已冻结为只读归档**——历史 40 条已全部迁入新箱（逐条 `legacy:` 令牌）；**勿再写入**；3 天双通道对账至 **09-16**（`Code` 每日核对）。
> 历史条目（≤2026-09-10）在 `docs/HANDOFF_ARCHIVE.md`。

## 🔴 迁移与对账状态（2026-09-13）

- **切换执行**（用户批准，`Code`）：旧箱 **40 条**（code 23 / paper 3 / perf 14 / baseline 0）已导入新箱（对账 + 抽样逐字全过）；旧 `docs/handoff/*` **即冻结只读**。
- **切换公告**（新箱）：`mb-20260913-095657-code-5655`→Paper ｜ `mb-20260913-095657-code-98fb`→Baseline ｜ `mb-20260913-095657-code-148c`→Perf（要求：新通道回执 + 各工作区自行同步 M7）。
- **对账期**：至 **2026-09-16**（`Code` 每日核对旧箱无新写入）；此后本目录仅为历史归档。
- **M7**：各 Agent 自行同步（更新本工作区 `copilot-instructions.md` 指针 + 自注册 MCP，见新仓 `mcp.example.json`）。

## 🔴 活跃事项速览（切换前快照 · 仅存史）

- **→ Paper ⬜**：k×N 热力图（10 张）是否修复版重跑（问询）；E-3/E-4/E-5 修复版交付已发；**主表 50/50 完成（09-13，数字已发信箱/PROGRESS）**
- **→ Paper/Code 🔄**：补充实验**已排程激活**（09-13；ⓐ2+ⓑ1+ⓓ15 → tasks #16–#33，E-2 完自动接跑，双卡 ~0.8–1.2 天）；ⓒ 旧码对照→**历史结果替代**（口径提醒已发；旧 5 种子 RT 0.7692/RB 0.7845）；k×N 保留+协议注
- **→ Baseline ⬜**：①接入汇报 ②第二波方法评估（R1-4）③服务器硬约束
- **→ Perf 🔄**：①真实数据复测 ✅；②M4 方案审批①–④落定；③**部署/构建/门 1/2 闭环**：通道 ✅ / 环境 ✅ / **`__abi__` 就位**（新 wheel sha `1ffe3216…`；契约 `k1k2-v2-2026-09-11` ✓）/ **第 1 批全绿复验**（cargo 21/21、check_accel 12/12、pytest 22、上游 1800+908 全 PASS、bench 14/14；K1 7–11×、K2 56–296×）；**适配器**：✅ 落地+服务端双短跑（⑦）**PASS**（`abfe664` 已 pull；默认启用；逐位一致：端到端 48.9→32.8s = **1.49×**、训练 1.42×、推理 1.76×）→ **剩余**：⑧ profile（可选）

## 全局备注

- 工作区分布：`Paper`=`D:\Sign_DygFormer` · `Code`=本仓库 · `Baseline`=`D:\codes\DynamiSE_DySDGNN_repro` · `Perf`=`D:\codes\SignDyG-Perf`（同机绝对路径互见；改动请在 DyGLib 仓库内提交，或由 `Code` 代提交）。
- **跨工作区纪律（2026-09-12 用户原则）**：各 Agent 只编辑**自己工作区**的文件（信箱除外）；跨工作区改动请求对方执行，勿代改。
- **服务器纪律（2026-09-11）**：服务器访问**唯一通道 = `Code`**，且须用户**逐次明确许可**；`Paper`/`Baseline`/`Perf` 一律禁止接触服务器（含自动运行 agent）。
- **本地网络代理（2026-09-12，用户提供）**：`http://192.168.10.3:7897`（Clash）——供各 agent 访问 GitHub 等外部资源（Code 实测 GitHub API 200 ✓）；仅网络访问，**不改变服务器纪律**。
- 代号 ↔ 历史别名：`Paper`=A · `Code`=B · `Baseline`=C · `Perf`（无旧名）；详见 `docs/AGENTS_REGISTRY.md`。
- 状态由**发件人**维护（新箱：`set-status`）；协议第 3 步不变；收件方以「回执行」反馈处理结果。
