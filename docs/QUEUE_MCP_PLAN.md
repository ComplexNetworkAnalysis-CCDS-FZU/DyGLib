# Queue MCP 设计与实施计划（草案 v0.1）

> **提出**：用户 2026-09-26（任务编排 MCP 化）；**架构选型**：**A = stdio-over-SSH**（用户倾向）。
> **状态**：草案——待用户对 §8 开放问题拍板后进入实现；建议实现窗口 **9-30 检查点之后**（只读切片可随时先做）。
> **作者**：`Code`。背景动因：2026-09-25/26 深夜队列三次重排（网格⇄E2 位置调整）全部靠"手工构建区间替换文件 + 回读"，状态查询靠临时探针脚本——需要 typed、带守卫、带审计的编排接口。

---

## 1. 目标 / 非目标

**目标**
- 把服务器任务队列（`tools/queue/tasks.txt` + 派发指针 + 运行状态）暴露为 **MCP 工具**：typed 操作、指针守卫、原子写、自动备份、回读校验、审计留痕、一键状态。
- 消除"手工 build 替换文件 / 手写 `_state_*.sh` 探针"两类重复劳动与人为差错面。

**非目标（本轮）**
- 不改 `queue_daemon.sh` 的派发契约（唯一消费者保持现状）。
- 不做 Web UI、不做多用户鉴权体系（访问控制=ssh 凭据 + 配置分发）。
- 不做 kill/取消能力（危险操作，见 §8-Q4）。

## 2. 架构（A：stdio-over-SSH）

```
[VS Code / 各 Agent] ── stdio(MCP) ──> ssh(非交互) ──> [服务器] queue_mcp_server.py
                                                          ├─ queue_core.py  （共享核心）
                                                          ├─ tasks.txt / running.txt / 备份 / 审计
                                                          └─ ps / nvidia-smi / saved_results（只读状态）
```

要点：
- **服务端代码放在 DyGLib 仓内**（`tools/queue/mcp/`），随既有 git 部署纪律同步（push→pull→校验），不引入新传输通道。
- **零新增端口**：stdio 管道穿过 ssh；`ssh -o BatchMode=yes` 防交互卡死。
- **stdout 专供 MCP 协议**；一切日志走 stderr（或落文件）。
- MCP 进程**无常驻**：VS Code 按会话拉起/回收（无状态、无锁文件、服务器重启无影响）。

客户端配置（各 workspace `.vscode/mcp.json`，M7 约定各 Agent 自注册；实现时以 VS Code 当前 schema 校验）：

```jsonc
{
  "servers": {
    "queue": {
      "type": "stdio",
      "command": "ssh",
      "args": [
        "-o", "BatchMode=yes",
        "fedsa@172.17.173.102",
        "cd ~/DyGLib && ~/queue-mcp-venv/bin/python -m tools.queue.mcp.server"
      ]
    }
  }
}
```

## 3. 模块与目录

| 文件 | 职责 |
|---|---|
| `tools/queue/queue_core.py` | **共享核心库**：读/写 tasks.txt、指针读取（与 daemon 同口径）、备份、原子替换（temp+fsync+`os.replace`）、回读校验、审计 JSONL、守卫（拒绝编辑 ≤ 指针的行）、status 采集 |
| `tools/queue/mcp/server.py` | FastMCP 薄壳：工具注册、参数校验、错误语义（把 core 的拒绝原因原样返回） |
| `tools/queue/edit_remote_tasks.py` | **重构为调用 queue_core**（CLI 保留为后备通道，行为不变） |
| `tools/queue/audit/` | 审计日志 + 编辑器备份（**服务器本地，不入仓**） |
| `tools/queue/tests/` | 单元测试（本地 temp 目录模拟服务器布局，不触真队列） |

## 4. 工具清单（分阶段）

**阶段 1 — 只读（零风险，可先做）**
- `queue_status()` → 指针（`len(running.txt)`）、下一派发行预览、在跑进程（argv 解析出 script/dataset/seed/gpu）、GPU 快照、各批计数（按 `saved_results` 文件名 tag：B2/B4/B5/E2/GRID/LF30…）、近期完成
- `queue_show(a, b)` / `queue_tail(n)` / `queue_audit(n)`（备份+指纹列表）

**阶段 2 — 写（仅 `Code` 分发；统一契约见下）**
- `queue_insert(after, lines)`
- `queue_replace_range(a, b, lines)`（现工具同款）
- `queue_move_block(a, b, after)`（typed 移动块——本次三连重排的根因场景）
- `queue_append(lines)`

**统一写契约**：① `a/b/after > 指针`（否则拒绝并回传当前指针与安全区间）；② 备份先行（`tasks.txt.bak-<ts>`，返回 `backup_id`）；③ 原子替换；④ 回读逐行校验；⑤ 审计一条；⑥ 返回 `{ok, total, backup_id, sha256, preview}`；⑦ 可选 `expected_sha256`（乐观并发：指纹不符即拒绝）。

**阶段 3 — 可选**
- `queue_undo(backup_id)`（带指针守卫的恢复）；`batch_generate(kind,…)`（网格/LF30/E2 模板生成）；服务端 `--read-only` 开关（供只读档分发）。

## 5. 安全模型

- **指针纪律（daemon 契约）**：派发行号 = `running.txt` 非空行数 + 1。⚠️ 实现第一步：**重读 `queue_daemon.sh` 固化"非空行"的精确口径**（空行计数、CRLF），并写 parity 测试。
- **原子性**：写 `tasks.txt.tmp` → fsync → `os.replace`（读者永不看到半行）；备份在替换前落盘。
- **审计**：`audit/queue_audit.jsonl`：`{ts, actor, op, args_digest, before_sha256, after_sha256, backup_id}`（actor 默认 `code`，可参数覆盖）。
- **访问控制**：写档配置只分发 `Code`；只读档（`--read-only`）可分发 `Paper`/`Baseline`/`Perf`（见 §8-Q2）。
- **幂等**（可选，学 mailbox）：客户端 `token` 去重，防重放。

## 6. 部署与运维

- **Python 环境**：服务器专用 venv（建议 `~/queue-mcp-venv`，`pip install mcp` 锁版本）——**不动 `gc` 环境**（见 §8-Q3）。
- 部署流程沿用纪律：`git push` → 服务器 `git pull --ff-only` → `git log -1` + 关键标记 grep + `python -c "import tools.queue.mcp.server"` 冒烟（前台同步执行）。
- 故障模式：ssh 不通 → 工具报错（不阻塞其它工作）；MCP 进程崩溃 → VS Code 自动重启会话；服务器重启 → 无影响（无状态）。
- 回退：`edit_remote_tasks.py` CLI 保留为后备通道；两者共享 `queue_core`，不会行为漂移。

## 7. 测试与验收

- **单元**（本地，temp 目录）：守卫（≤指针拒绝）、`replace_range` 与现 CLI **parity**、备份/回读、审计、原子替换。
- **Golden 用例**：用 2026-09-25/26 三个真实操作（网格→B4后；恢复原排；E2 前移）做逐行对拍。
- **集成**（服务器）：`QUEUE_TASKS_PATH` 环境变量指向 **sandbox 副本** 全工具烟测；真队列只跑 `queue_status`（只读）。
- **验收**：MCP 重演三个 golden 操作，结果与已知良好队列逐行一致。

## 8. 开放问题（待用户意见）

| # | 问题 | 我的建议 |
|---|---|---|
| Q1 | **排期**：9-30 检查点后实现？还是只读版先做？ | 只读版可随时（1–2h，零风险）；写版排 9-30 后 |
| Q2 | **只读档**是否分发给 `Paper`/`Baseline`/`Perf`（他们自查队列状态，省去问 Code）？ | 建议开（只读零风险；仍 M7 自注册） |
| Q3 | 服务器 **venv**（`~/queue-mcp-venv`）还是直装 `gc`？ | 专用 venv（不污染实验环境） |
| Q4 | 是否需要 **kill/取消** 能力？ | 默认不做（如需，另立设计）|
| Q5 | 审计 JSONL 是否随 git 入仓？ | 否（仅服务器本地；摘要入 PROGRESS）|
| Q6 | 目录：`tools/queue/mcp/` 子包（推荐）还是平铺？ | 子包 |
| Q7 | MVP 写集是否包含 `undo`/`batch_generate`？ | 不含（阶段 3）；先 CRUD + move |

## 9. 工作量估算（专注工时）

| 项 | 估计 |
|---|---|
| `queue_core` 抽取 + CLI 重构（parity 通过） | 2–3h |
| MCP 薄壳 + 工具 schema + 错误语义 | 1.5–2h |
| 测试（单元 + golden + 集成烟测） | 1.5–2h |
| 服务器部署（venv、git 通道、校验）+ 配置片段与说明 | 1–1.5h |
| **合计** | **≈ 1 天**（建议 9-30 后；只读切片 ≈1–2h 可先行） |

## 10. 交付物

1. `tools/queue/queue_core.py`、`tools/queue/mcp/server.py`、重构后的 `edit_remote_tasks.py`
2. `tools/queue/tests/`（parity/golden/烟测）
3. `docs/QUEUE_MCP.md`（用户手册：工具语义、指针纪律、故障处理）+ `.vscode/mcp.json` 配置片段（写档/只读档各一份）
4. PROGRESS 条目 + tools/README 登记
