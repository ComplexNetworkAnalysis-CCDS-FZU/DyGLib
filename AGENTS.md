# SignDyG 修订协作契约（NEUCOM-D-26-13975）

多 Agent 协作修订。**2026-09-11 起启用角色代号**：`Paper`=论文、`Code`=代码/实验、`Baseline`=基线复现、`Perf`=Rust+PyO3 加速、`DirG`=有向图（2026-09-22）。会话工作区 = 本仓库时，本文件自动加载。
分工（互不越界）：**Paper**（原 A）= 论文正文/理论/回复信（工作区 `D:\Sign_DygFormer`）；**Code**（原 B）= 代码/实验/数据/服务器（本仓库）；**Baseline**（原 C）= DynamiSE/DySDGNN 基线复现（工作区 `D:\codes\DynamiSE_DySDGNN_repro`，R2-5 方案①；⛔ 禁服务器）；**Perf**（新）= Rust+PyO3 内核加速（路线 C；工作区 `D:\codes\SignDyG-Perf`；⛔ 禁服务器）；**DirG**（2026-09-22 新）= 毕业论文第二点·有向图（有向符号图方法；工作区 `D:\codes\SignDyG-Dir`；首单 `docs/DIRG_KICKOFF.md`）。
→ **服务器纪律（2026-09-22 修订）**：**服务器访问主通道 = `Code`**，须**用户逐次明确许可**；**特例：`DirG` 经用户逐次明确许可可直连服务器**（新角色专项批准）；其余 Agent（`Paper`/`Baseline`/`Perf`，含自动运行）一律禁止接触服务器。
→ **统一信箱（2026-09-13 起）**：`D:\codes\agent-mailbox`（唯一正式通道；只经 CLI/MCP 工具写；旧 `docs/handoff/*` 冻结只读；M7 = 各 Agent 自行同步指针与 MCP）。
→ 身份/职责唯一权威：`docs/AGENTS_REGISTRY.md`（新 Agent 先登记再开工）。

## 权威层级
- ★ 所有决策以 `docs/ADVISOR_DECISIONS.md` 为准，冲突时以此为准。
- `docs/AGENTS_REGISTRY.md` Agent 注册表（身份/职责唯一权威）；`docs/REVIEWER_COMMENTS.md` 审稿归档；`docs/EXPERIMENT_PLAN.md` 实验计划；`docs/PROGRESS.md` 进度与结果。

## 开工必读 / 收工必写
1. 会话开始：读 `docs/HANDOFF.md`（**指针页**）→ **统一信箱** `D:\codes\agent-mailbox`：`python -m mailbox.cli list --for <自己> --open`（或渲染视图 `handoff/outbox-<自己>.md`）→ `docs/ADVISOR_DECISIONS.md` → 相关 `docs/PROGRESS.md`。
2. 任务结束：更新 `docs/PROGRESS.md`；交付/请求**只经工具**写入新信箱自己箱（`send --from <角色> --to <对方> --type … --text …`；回执 `ack`；状态由原发件人 `set-status`，⬜→🔄→✅）；收件方处理后在**自己箱内发回执**，发件人把状态改 ✅（勿编辑他人文件）。
3. `Paper` / `Baseline` / `Perf` / `DirG` 在其他工作区：以绝对路径 `D:\codes\DyGLib\docs\` 直接读写（同机即时互见）；**M7（2026-09-13）各 Agent 自行同步**：更新本工作区根 `copilot-instructions.md` 的信箱指针（新路径 + 只经工具写）+ 自注册 MCP（按新仓 `mcp.example.json`）。

## 修改纪律
- 涉及代码/实验/数据的修改须经用户批准后执行。
- **跨工作区边界（2026-09-12 用户原则）**：各 Agent 只编辑**自己工作区**的非信箱文件；需要他人工作区改动时，通过信箱请求对方执行，勿代改。
- `Code` 的实验结果以「表格 + 结论」写入 `docs/PROGRESS.md`，供 `Paper` 直接引用到论文；`Baseline` 的复现结果以 summary（含协议说明）交付并在 PROGRESS 登记。

## 目录约定（2026-09-11 起）
- **辅助脚本 / 验证脚本 / 临时验证代码一律放 `tools/`**（`tools/queue/` 队列、`tools/verify/` 验证），不得散落仓库根目录；正式工具需登记在 `tools/README.md`，并在**仓库根目录**运行（脚本内用相对路径）。
- 服务器同步一律走 **git**（`push` → 服务器 `git pull --ff-only`）；禁止用 scp 手工传文件（不可追溯）。
- 部署类操作必须在**前台同步执行并校验**（`git log -1` + 关键标记 grep），禁止放入 `& disown` 后台链或管道（会静默失败）。
