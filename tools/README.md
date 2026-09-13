# tools/ — 辅助脚本统一目录（约定见下）

> **约定（2026-09-11 起，用户指示）**：所有**辅助脚本、验证脚本、临时验证代码**一律放在 `tools/` 下，**不得散落在仓库根目录**。
> 运行方式：**在仓库根目录执行**（脚本内的相对路径如 `processed_data/`、`results/` 均相对根目录），例如：
> `python tools/verify/verify_ras_rae_leak.py`
> 一次性临时脚本用完请删除或在此登记；正式工具请在本文件登记用途与用法。

## 目录结构

| 路径 | 用途 |
|---|---|
| `tools/queue/` | 实验队列守护进程（自动分配空闲 GPU） |
| `tools/verify/` | 验证脚本（单测式断言、数据统计、结果核对） |
| `tools/stats/` | （预留）统计汇总类脚本（如 p 值/显著性） |

## 已登记脚本

### tools/queue/
- `queue_daemon.sh` — 队列守护进程。**用法**：把待跑命令逐行写入 `tasks.txt`——① `run_experiments.py` 参数（**不含 `-g`**，daemon 自动补）；② **以 `@` 开头的原生命令**（用 `@GPU@` 占位显卡号，用于非 run_experiments 任务，如 E-4 TD）。daemon 每 60s 探测 GPU 空闲（lock 文件 PID + `nvidia-smi` 显存 > 500MiB 判忙）并串行推进。
  - 日志：`tools/queue/queue.log`（调度轨迹：idle/waiting/dispatch）、`tools/queue/logs/task_<n>.log`（各任务输出）
  - 状态：`tools/queue/running.txt`（已分发任务行号）、`/tmp/gpulock.<gpu>`
  - 停止：`pkill -f queue_daemon.sh`
- `tasks.txt` — 任务清单（每行一条，按顺序分发）。
- `selftest.sh` — **自检脚本**（不启动任务）：校验任务清单解析、GPU 忙闲探测（lock + 显存）、选卡函数。用法：`bash tools/queue/selftest.sh`。

### tools/verify/
- `verify_ras_rae_leak.py` — **RAS/RAE/泄漏修复的三项断言**：① 翻转 pos0 标签后 BTE 输出不变（泄漏已封）；② RAE on ≠ off（direct 证据真实生效）；③ indirect 只落在真第三方位置。
- `check_archived_results.py` — 打印本地归档结果（`results/E-2_ablation/raw/`）的指标/耗时指纹，用于与服务器新结果对照、判定文件是否被新 run 覆盖。
- `repeat_sign_stats.py` — 统计各数据集**重复交互率**与**重复对符号翻转率**（用于解释 RAS/RAE 的作用面；BA/OTC 重复率 ~80% 但翻转率仅 2.5%，WikiVote 翻转率 33% 但重复率仅 6.9%）。
- `cn_quirk_analysis.py` — **CN 伪交集怪癖分析**（玩具探针 + 真实数据量化：伪 CN 查询占比/采样输出差异率）；结论见 `docs/ANALYSIS_CN_PSEUDO_INTERSECT.md`。用法：`python tools/verify/cn_quirk_analysis.py [--queries 2000]`。
- `snapshot_results.py` — **结果快照**：从本地 `results/**/raw/`（已同步归档）快速汇总 E-3/E-4/E-5sign/主表 sign 各组指标（auc/ap/sign_f1 明细 + 5 种子 mean±pstd）。用法：`python tools/verify/snapshot_results.py`（不访问服务器）。
- `split_handoff.py` — **一次性迁移（2026-09-12）**：`docs/HANDOFF.md` 单文件信箱 → `docs/handoff/outbox-*.md`「一人一箱」（按节+发出方路由；条目内容逐字保留，仅把「发出方」列改为「收件人」列）。
- `test_accel_seam.py` — **M4 加速接缝自检**：① CLI 开关（默认启用 / `--no-accel` / `--accel` 显式 / `SIGNDYG_ACCEL` 兜底；子进程隔离）；② K1/K2 原路径 vs 加速路径**逐位一致**（Perf 参考实现桩 + 本机真实内核复跑）。用法：`python tools/verify/test_accel_seam.py`（本地运行，不触服务器）。
- `probe_k2_binding_dtypes.py` — **K2 Rust 绑定 dtype 契约探测**：记录 `bte_sign_effect` 的 accept/reject 矩阵（nodes / padded ids=int64、signs=int8、times=float32；query float64 或 None 均可；不符 = fail-fast TypeError）。用法：`python tools/verify/probe_k2_binding_dtypes.py`。
- `compare_accel_runs.py` — **两次训练结果 JSON 对照**（排除计时/显存/`accel` 字段；退出码 0 = 除计时外逐位一致）。用法：`python tools/verify/compare_accel_runs.py <a.json> <b.json>`。
- `cn_microbench.py` — **CN 共现编码细粒度微基准（K3 前置；纯 CPU，本机可跑）**：逐位复刻 `count_nodes_appearances` 并在 12 个子步骤打点（np.unique / 逐元素 `apply_` / stack / 掩码…，B=200、L≈40 口径对齐 BA 主实验），另含全向量化原型（仅评估，不落库）与逐位一致性校验。**BA 口径结论**：本机原实现 ≈37–41ms/次，逐行固定成本占 67–81%（np.unique×2 占 37–42%）、逐元素 apply_ 占 14–29%；**向量化原型 28×（L=40；14–55×@L16–100）且逐位一致**。用法：`python tools/verify/cn_microbench.py [--sweep] [--dist uniform] [--out …]`；产物 JSON：`cn_microbench_result.json` / `cn_microbench_sweep.json`。

### tools/sync/
- `fetch_results.py` — **结果归档同步**（只读服务器，ssh 读取，不 scp）：把服务器已完成的实验 JSON 拉取到本地 `results/**/raw/`，保持归档与服务器一致；输出 sha256 清单并追加 `results/_sync_raw_log.csv`。用法：`python tools/sync/fetch_results.py --set e2|e3|e4|e5|main-a|main-b|main-c|all`（`main-*` = 主表分批次；其余批次随完成逐步新增）。**服务器访问须用户逐次明确许可。**

## 待议（尚未迁移的既有脚本）

以下为先于本约定存在、散落其他位置的脚本，是否一并迁入 `tools/` 需与用户确认（迁移需同步更新 docs 引用）：
- `server_setup.sh`（根目录，服务器一次性环境准备）
- `result_collect.py`（根目录，结果收集）
- `dataset_analysis/compute_stats.py`（结果统计：mean±std、配对 t 检验；docs 中有引用）
- `analysis/*.py`、`dataset_analysis/*.py`、`fig/*.py`（早期数据分析/绘图脚本）
