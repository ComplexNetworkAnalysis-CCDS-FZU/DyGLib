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
| `tools/server_setup/` | 服务器端构建脚本（env 构建等；经 git 通道部署后在服务器 nohup 执行） |
| `tools/stats/` | （预留）统计汇总类脚本（如 p 值/显著性） |

## 已登记脚本

### tools/queue/
- `queue_daemon.sh` — 队列守护进程。**用法**：把待跑命令逐行写入 `tasks.txt`——① `run_experiments.py` 参数（**不含 `-g`**，daemon 自动补）；② **以 `@` 开头的原生命令**（用 `@GPU@` 占位显卡号，用于非 run_experiments 任务，如 E-4 TD）。daemon 每 60s 探测 GPU 空闲（lock 文件 PID + `nvidia-smi` 显存 > 500MiB 判忙）并串行推进。
  - 日志：`tools/queue/queue.log`（调度轨迹：idle/waiting/dispatch）、`tools/queue/logs/task_<n>.log`（各任务输出）
  - 状态：`tools/queue/running.txt`（已分发任务行号）、`/tmp/gpulock.<gpu>`
  - 停止：`pkill -f queue_daemon.sh`
- `tasks.txt` — 任务清单（每行一条，按顺序分发）。
- `selftest.sh` — **自检脚本**（不启动任务）：校验任务清单解析、GPU 忙闲探测（lock + 显存）、选卡函数。用法：`bash tools/queue/selftest.sh`。

### tools/server_setup/
- `build_env_dygmamba.sh` — **R1-4 波二 DyG-Mamba 独立 env 构建**（服务器端）：torch2.1.0(+cu118) → **v2：pin numpy<2/setuptools69** → nvcc（conda cuda-nvcc=11.8）下源码构建 `causal-conv1d==1.4.0 / mamba-ssm==2.2.2`（arch=7.5）→ CUDA 算子自检。用法（服务器）：`cd ~/DyGLib && nohup bash tools/server_setup/build_env_dygmamba.sh > ~/envbuild_dygmamba.log 2>&1 &`
- `build_env_scadyg.sh` — **R1-4 波二 ScaDyG 独立 env 构建**（旧栈）：**v2：重建 env** + torch1.12.1+cu116 / numpy1.23.4 / **PyG 精确轮子 pin**（scatter2.1.0+pt112cu116 / sparse0.6.16 / cluster1.6.0 / spline-conv1.2.1）/ dgl1.0.0 / deepsnap / py-tgb → 自检。用法同上（日志 `~/envbuild_scadyg.log`）。

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
- `test_cn_vec_seam.py` — **CN 向量化接缝自检（numpy 过渡版，2026-09-14 落地；受 accel 开关控制）**：① accel 路径（`utils/accel.py::cn_counts_vec`）与逐行原路径**逐位一致**（随机重复/全零/单行/L=1/**Ls≠Ld**/大 id/全同值/空列/单侧空）；② 开关路由（`accel.on=True` 走向量化、`False` 回原路径，spy 计数断言）；③ 输出契约（float32、(B,L,2)）；另验 B=0 安全返回。用法：`python tools/verify/test_cn_vec_seam.py`（本机/服务器均可，纯 CPU；手动注入 `accel.on`，不动内核/环境解析）。
- `cn_bincount_bench.py` — **CN 重构评估（2026-09-14，用户要求）**：bincount 直方图版（行复合键 O(N) 直方图 + 四次查表，无排序/searchsorted）对照现落库 unique+searchsorted 版；扫 n_nodes/B/L/分布 + 边界自检。**结论：bincount 不如现行**（BA 口径 1.29ms→3.24ms=0.40×；全配置中位 0.42×，仅 B=18 尾批 1.73×）；直方图域=B·n_nodes，大 id 时爆内存（域膨胀缺陷，真实数据不触发）。用法：`python tools/verify/cn_bincount_bench.py [--iters 30]`；JSON：`cn_bincount_bench_result.json`。
- `nh5_report.py` — **第 2 批补充集指标汇总打印**（胜者/复核 5 种子 mean±std、OTC/WV 邻域、RT 探边 + 主表现行对照；含**同 seed 配对比较**：Δmean、配对 t，df=4 时 |t|>2.78 ⇔ p<0.05）。用法：`python tools/verify/nh5_report.py`（需已完成 `--set nh5` 同步）。- `bte_signal_check.py` — **BTE 离线信号体检（2026-09-14；纯 CPU/轻量/不训模型）**：在测试期真实边上直接检验「平衡理论三元证据」与真实符号的关联（忠实复刻 BTE 计数：pos/neg 出现对计数 + 去重投票；对照多数类与直接历史预测器；**乐观上界口径**：全历史截近 K 求交，模型侧另有 CNAS 窗口 + NN 截断）。用法：`python tools/verify/bte_signal_check.py [--edges 1200] [--hist-cap 100] [--datasets …]`（秒级–分钟级，电池机可跑）。
- `edit_remote_tasks.py` — **服务器队列安全编辑器（零 sed/零转义）**：ssh+python+base64 回写 `tools/queue/tasks.txt`；支持 `--show a,b` 查行、`--insert-after N --lines-file F` 插行；写入前自动备份（`tasks.txt.bak-<时间戳>`）、写入后回读逐行校验。背景：2026-09-14 经 PowerShell→ssh 的 sed 转义被吃掉、毁过三行任务。用法：`python tools/queue/edit_remote_tasks.py --show 77,84`。
- `metrics_table.py` — **指标速览**：打印若干结果 JSON 的紧凑全指标表（默认 auc/ap/sign_f1/f1_mac/f1_wt/f1_mic/acc；标签取文件名 `LF-Best.`→`.P1.` 之间的模块旗标）。用法：`python tools/verify/metrics_table.py <globs...> [--keys …]`（消融/对照快速比对用）。
- `agg_configs.py` — **按配置旗标聚合 + 同种子配对**：扫描目录内结果 JSON（解析文件名 `seed{N}` + 旗标 `RAS-[ED].RASE-[ED].BTE-[ED].CNAS-[ED]`），输出每配置 mean±pstd（ddof=0），`--pair A B` 打印 A−B 配对差值（Δmean/sd/t）。用法：`python tools/verify/agg_configs.py \"results/E-2_ablation/raw_seeds/*\" [--pair TAG_A TAG_B]`（2×2 消融分析用）。\n- `cnas_last_cn_stats.py` — **CNAS 窗口结构离线统计（纯 CPU）**：验证「CNAS 加密→recent-N」退化的结构性残余——last-CN 截断（~15% 最新事件恒被丢弃）、窗口并集覆盖率（@LF/@20）、no-CN 率。用法：`python tools/verify/cnas_last_cn_stats.py --edges 2000`。
### tools/sync/
- `fetch_results.py` — **结果归档同步**（只读服务器，ssh 读取，不 scp）：把服务器已完成的实验 JSON 拉取到本地 `results/**/raw/`，保持归档与服务器一致；输出 sha256 清单并追加 `results/_sync_raw_log.csv`。用法：`python tools/sync/fetch_results.py --set e2|e2b|e2d|e3|e3x|e4|e5|nh5|base-gpu|main-a|main-b|main-c|main-d|main-e|all`（`main-*` = 主表分批次；`nh5` = 第 2 批补充集（胜者 5 种子 + 复核 5 种子/OTC·WV 邻域/RT 探边，32）；`e2s` = ⓑ 5 种子加深（full/vanilla/CNAS-only 各 25）；`e2c` = E-2c 5 种子（BTE-only/base 各 25 + RT 的 +RAS/+RAE 各 5）；`base-gpu` = Baseline GPU 产物（30 JSON + 2 summary CSV → `results/baseline_m5/`）；**硬校验数量**，不符即中止）。**服务器访问须用户逐次明确许可。**

### tools/fig/
- `gen_p_patch_heatmap.py` — **R2-11 P 敏感性热力图（2026-09-14）**：读 `results/E-3_patch/raw/`（E-3 v3 CN 修复版，linksign、seed42、NN/LF-Best、TE；只读）绘制 **5 数据集 × P{1,3,5,7}** 双面板热力图（AUC 主面板 + $F1_{wt}$ 副面板），运行时与 `E3_patch_summary.md` 做 84/84 交叉校验；**配色与布局对齐仓库既有热力图**（`analysis/param-graph.py`：seaborn `cmap="crest"`、`annot/fmt=".4f"`、显示名映射 WikiRfA/RedditTitle/RedditBody、轴标签 14pt）。产物：`figures/fig_p_patch_heatmap.{png,pdf,csv}`（PNG 300dpi）。用法：`python tools/fig/gen_p_patch_heatmap.py`（仓库根目录运行，不触服务器）。**注：图产物（png/pdf/csv）不入库**（用户 2026-09-14 指示：二进制不便管理）——按需本地重新生成即可（脚本确定性，仅依赖已归档 raw JSON）。

## 待议（尚未迁移的既有脚本）

以下为先于本约定存在、散落其他位置的脚本，是否一并迁入 `tools/` 需与用户确认（迁移需同步更新 docs 引用）：
- `server_setup.sh`（根目录，服务器一次性环境准备）
- `result_collect.py`（根目录，结果收集）
- `dataset_analysis/compute_stats.py`（结果统计：mean±std、配对 t 检验；docs 中有引用）
- `analysis/*.py`、`dataset_analysis/*.py`、`fig/*.py`（早期数据分析/绘图脚本）
