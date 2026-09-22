# SignDyG-Dir 协作契约（`DirG` · 毕业论文第二点·有向图）

> 本工作区（`D:\codes\SignDyG-Dir`）会话自动加载。**开工先读** `D:\codes\DyGLib\docs\DIRG_KICKOFF.md`（首单任务包）；本文件之下保留 DyGLib 主契约全文供参考（引用时以其最新版为准）。

## 身份与职责
- 代号 `DirG`（有向图）= **毕业论文第二点全权负责**：任务定义 / 代码 / 实验 / 数据 / 交付。
- 登记：DyGLib `docs/AGENTS_REGISTRY.md`（身份/职责唯一权威）；决策：DyGLib `docs/ADVISOR_DECISIONS.md`（★ 冲突以此为准）。
- 范围（2026-09-22 用户拍板）：**有向符号图方法**——方向感知 in/out 序列 + Status Theory 有向 2-路径（复用 `models/DirectSignDyGFormer.py` / `models/DirectStatusEncoder.py` / `utils/direct_neighbor_sampler.py`）；数据集 BA/OTC（天然有向）+ RT/RB/WV（主线口径），myket 备用。

## 边界（跨工作区原则 2026-09-12 用户）
- 只编辑**本工作区**的非信箱文件；DyGLib 主线 `D:\codes\DyGLib` **只读**（需要其改动 → 信箱请求 `Code`，勿代改）。
- ⛔ 不推 `sign-adoption`；不动既有主表产物；不改他人工作区；信箱 `D:\codes\agent-mailbox` **只经 CLI/MCP 工具**读写。

## 服务器纪律（2026-09-22 用户专项批准）
- 可直连 `fedsa@172.17.173.102`（`~/DyGLib`，conda `gc`），但**每次操作须用户逐次明确许可**。
- GPU 窗口由 `Code` 协调（**当前修订 ours 批次优先**）；不抢卡、不覆盖他人产物。
- 同步走 **git**（禁 scp）；部署类操作前台同步执行并校验；杀任务"**先子后父**"；手工 `@` 行含 `DATASET_EXTRA`（RT/RB/WV 需 `--tail-num 20000`）。

## 信箱与数据
- 信箱 `D:\codes\agent-mailbox`；MCP 已配置（`.vscode/mcp.json`，`MAILBOX_AGENT=DirG`）；开工 `list --for DirG --open`；收工更新本仓 `REPORTS/` + 对外发信（`Code` 登记 DyGLib `PROGRESS.md`）。
- `processed_data/`、`DG_data/` = junction 至 DyGLib（**只读**）；新预处理数据放本仓 `preprocess_data/`。

## 实验纪律
- 先护栏/单测后实验；单变量；代际标记（如 `.DR-*`）；默认关 = 零行为变更；判据 Δ≥0.005 + 同种子配对 + ≥3/5 同向。
- 涉及代码/实验/数据的修改**须经用户批准后执行**；交付 = 表格 + 结论（mean±pstd + 配对 Δ/t/p）。

---

# SignDyG 修订协作契约（NEUCOM-D-26-13975）（DyGLib 主契约·参考全文）

多 Agent 协作修订。**2026-09-11 起启用角色代号**：`Paper`=论文、`Code`=代码/实验、`Baseline`=基线复现、`Perf`=Rust+PyO3 加速。会话工作区 = 本仓库时，本文件自动加载。
分工（互不越界）：**Paper**（原 A）= 论文正文/理论/回复信（工作区 `D:\Sign_DygFormer`）；**Code**（原 B）= 代码/实验/数据/服务器（本仓库）；**Baseline**（原 C）= DynamiSE/DySDGNN 基线复现（工作区 `D:\codes\DynamiSE_DySDGNN_repro`，R2-5 方案①；⛔ 禁服务器）；**Perf**（新）= Rust+PyO3 内核加速（路线 C；工作区 `D:\codes\SignDyG-Perf`；⛔ 禁服务器）。
→ **服务器纪律（2026-09-11）**：**服务器访问唯一通道 = `Code`**，且须**用户逐次明确许可**；其他 Agent（`Paper`/`Baseline`/`Perf`，含自动运行）一律禁止接触服务器。
→ **统一信箱（2026-09-13 起）**：`D:\codes\agent-mailbox`（唯一正式通道；只经 CLI/MCP 工具写；旧 `docs/handoff/*` 冻结只读；M7 = 各 Agent 自行同步指针与 MCP）。
→ 身份/职责唯一权威：`docs/AGENTS_REGISTRY.md`（新 Agent 先登记再开工）。

## 权威层级
- ★ 所有决策以 `docs/ADVISOR_DECISIONS.md` 为准，冲突时以此为准。
- `docs/AGENTS_REGISTRY.md` Agent 注册表（身份/职责唯一权威）；`docs/REVIEWER_COMMENTS.md` 审稿归档；`docs/EXPERIMENT_PLAN.md` 实验计划；`docs/PROGRESS.md` 进度与结果。

## 开工必读 / 收工必写
1. 会话开始：读 `docs/HANDOFF.md`（**指针页**）→ **统一信箱** `D:\codes\agent-mailbox`：`python -m mailbox.cli list --for <自己> --open`（或渲染视图 `handoff/outbox-<自己>.md`）→ `docs/ADVISOR_DECISIONS.md` → 相关 `docs/PROGRESS.md`。
2. 任务结束：更新 `docs/PROGRESS.md`；交付/请求**只经工具**写入新信箱自己箱（`send --from <角色> --to <对方> --type … --text …`；回执 `ack`；状态由原发件人 `set-status`，⬜→🔄→✅）；收件方处理后在**自己箱内发回执**，发件人把状态改 ✅（勿编辑他人文件）。
3. `Paper` / `Baseline` / `Perf` 在其他工作区：以绝对路径 `D:\codes\DyGLib\docs\` 直接读写（同机即时互见）；**M7（2026-09-13）各 Agent 自行同步**：更新本工作区根 `copilot-instructions.md` 的信箱指针（新路径 + 只经工具写）+ 自注册 MCP（按新仓 `mcp.example.json`）。

## 修改纪律
- 涉及代码/实验/数据的修改须经用户批准后执行。
- **跨工作区边界（2026-09-12 用户原则）**：各 Agent 只编辑**自己工作区**的非信箱文件；需要他人工作区改动时，通过信箱请求对方执行，勿代改。
- `Code` 的实验结果以「表格 + 结论」写入 `docs/PROGRESS.md`，供 `Paper` 直接引用到论文；`Baseline` 的复现结果以 summary（含协议说明）交付并在 PROGRESS 登记。

## 目录约定（2026-09-11 起）
- **辅助脚本 / 验证脚本 / 临时验证代码一律放 `tools/`**（`tools/queue/` 队列、`tools/verify/` 验证），不得散落仓库根目录；正式工具需登记在 `tools/README.md`，并在**仓库根目录**运行（脚本内用相对路径）。
- 服务器同步一律走 **git**（`push` → 服务器 `git pull --ff-only`）；禁止用 scp 手工传文件（不可追溯）。
- 部署类操作必须在**前台同步执行并校验**（`git log -1` + 关键标记 grep），禁止放入 `& disown` 后台链或管道（会静默失败）。
