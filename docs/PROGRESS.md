# SignDyG 修订实验进度记录（NEUCOM-D-26-13975）

> **维护者**：`Code`（代码与实验；历史代号 B）
> **用途**：所有 Agent（含 `Paper` 论文撰写）通过本文件了解最新进度。
> **规则**：每次实验状态变化立即更新本文件，并通过 git 同步（本地 push → 服务器 pull）。
> 提交截止：2026-10-10。

## 全局状态（2026-09-13 更新）

| 项目 | 状态 | 说明 |
|---|---|---|
| **Agent 代号启用** | ✅ 2026-09-11 | 协作体系改用角色代号：`Paper`/`Code`/`Baseline`/`Perf`（Rust 加速，新加入）；注册表 `docs/AGENTS_REGISTRY.md`；历史条目保留字母（A=Paper、B=Code、C=Baseline） |
| **Perf 工作区** | ✅ 2026-09-11 | `D:\codes\SignDyG-Perf`：DyGLib 热点内核**参考实现**（K1 采样链 / K2 BTE，numpy，含全部修复语义）+ 8 场景 golden fixtures + 上游逐位对照全绿 + 基线基准/回归脚手架；`TASKS.md`（M0–M5，双闸门：bit-exact + ≥10×）；⛔ **服务器禁区**（未经用户逐次许可禁止任何服务器接触，含自动 agent） |
| **Perf 进展** | ✅ 2026-09-12 | **M0–M3.6 全部完成**（09-11→12）：工具链（GNU）+ K1/K2 内核 + batch/GIL → **语义 v2**（CN 真交集；908 查询逐位 PASS）→ **真实数据口径复测**（BA/WV 只读派生；CN 实测 BA 1.39 / WV 9.11——合成「坍缩」为构造伪象；真实提速 **K1 4.4–5.6×（逐查询）/ 5.3–6.5×（batch）、K2 97–99×**；K1 1800 查询 + K2×2 逐位 PASS）。**M4 方案 v1.1 已交付**（审批①–④落定）；**部署就绪**：预检 + 独立仓通道 + Rust 环境（rustup/cargo 1.98.1、maturin 1.15.0）+ **首构建 ✅**；**第 1 批校验全绿 + `__abi__` 重建复验**（cargo 21/21、check_accel 12/12、pytest 22、上游 1800+908 全 PASS、bench 14/14；服务器口径 K1 7–11× / K2 56–296×；**门 1/2 闭环**）。详见 `docs/handoff/outbox-perf.md` 与 `outbox-code.md`（09-12 行） |
| **M4 加速适配器（Code）** | ✅ 2026-09-13 本地+服务端全绿 | `utils/accel.py` + K1/K2 接缝（sampler/BTE）+ CLI 开关：**默认启用**（用户口径；`--no-accel` 关闭 / `--accel` 显式 / `SIGNDYG_ACCEL` 环境变量兜底）；启用=fail-fast（未装/契约不符即报错，不静默回退）；结果 JSON 记录 `"accel"` 溯源。验证：接缝自检 **9/9**（参考桩+真实内核）、全量 BitcoinAlpha 冒烟**逐位一致**（CPU/L5/1ep：端到端 **1.33×**、训练 1.27×、推理 1.63×）；**服务端双短跑（⑦）PASS**（`abfe664` 已 pull；OFF/ON **逐位一致**；端到端 48.9→32.8s＝**1.49×**、训练 1.42×、推理 1.76×；CPU 口径/双卡任务并行中）。在跑批次后续子进程自动沿用新代码 |
| **CN 向量化（numpy 过渡版）** | ✅ 2026-09-14 本地落地（用户批准） | `utils/accel.py::cn_counts_vec()` + `NeighborInteractEncoder.count_nodes_appearances` 接缝，**受 accel 开关控制**（默认启用；`--no-accel` / `SIGNDYG_ACCEL=0` 回原路径；K3 Rust 到货后同接缝替换）。自检 `tools/verify/test_cn_vec_seam.py` 全绿（逐位一致含 Ls≠Ld/空行/大 id + 开关路由 spy + 契约）、`test_accel_seam.py` 回归全绿；本机 BA 口径 ~28×（`cn_microbench.py`）。**服务器 pull 推迟至 #34–#47 批次结束**（保持同代码口径；bit-exact 下指标等价、仅提速）→ 随后 pull + 服务器自检 |
| **Agent C 加入** | ✅ 2026-09-09 | `Baseline`（原 C）= DynamiSE/DySDGNN 复现（R2-5 ①），工作区 `D:\codes\DynamiSE_DySDGNN_repro`，权威 = `IMPLEMENTATION_SPEC.md`；结果登记 HANDOFF/PROGRESS |
| 服务器 | ✅ 可用 | 2026-09-01 恢复访问；**访问纪律（09-11）：唯一通道 = `Code`（须用户逐次许可），其他 agent（含自动）禁止接触** |
| 代码同步 | ✅ 完成 | 已推送 `sign-adoption` 至服务器裸仓库并 clone；**2026-09-13 晚批次**：M6 信箱切换 / E-2 v3 / P 补全 / CN 评估工具（`cn_microbench.py`）已提交推送，服务器 `pull --ff-only` 校验通过 |
| 数据就绪 | ✅ 完成 | `server_setup.sh` 已执行，WikiVote tail20000 已生成 |
| 环境安装 | ✅ 完成 | conda env `gc`（torch 2.2.2） |
| **GPU 驱动** | ✅ **已修复（2026-09-06）** | DKMS `nvidia/595.84` 已装（内核 6.8.0-124/138）；`torch.cuda.is_available()=True`，设备 `NVIDIA GeForce RTX 2080 SUPER`；fedsa 免 sudo 可访问。详见 `docs/GPU_DRIVER_INSTALL.md`（实际装 595.84，非计划 590）。**2026-09-13 深夜：重启对齐 NVML 至 595.91.07**（模块/用户态一致，`nvidia-smi` 恢复；启动内核 6.8.0-138） |
| 耗时预期 | ✅ 可切 GPU 口径 | 自 2026-09-06 起新实验可按 GPU 估算；E-3 及此前日志仍为 CPU 耗时 |
| **实验队列** | ✅ 运行中（15/15 已全部派发，09-13 04:46；剩 #14/#15 两批消融在跑） | `tools/queue/queue_daemon.sh` 常驻守护：探测空闲 GPU（lock PID + 显存）并自动派发 `tasks.txt` 中的任务；日志 `tools/queue/queue.log`；自检 `bash tools/queue/selftest.sh` 全部通过；**追加任务 = 往 `tasks.txt` 加一行**（不含 `-g`） |
| E-7 噪声 | ⛔ 本轮不做 | 模块 `utils/noise.py` 已实现保留 |

## 实验状态总览

| 实验 | 任务 | 数据集 | 状态 | 结果路径 / 备注 |
|---|---|---|---|---|
| E-1 效率 | sign | RedditTitle@20000 | ✅ 可提取 | E-5 sign seed42 已含 4 项效率数据，待汇总 |
| E-2 消融 | linksign | **全部 5 数据集** | ✅ **完成（v3 CN 真交集，09-13）**：v3 20/20 + ⓑ BTE-off 10/10 + ⓓ 网格 15/15 均已同步 | v3 汇总 `E2_ablation_summary.md`；ⓑ（单种子）：BTE 边际 BA≈0、OTC +0.0053、RT +0.0081，WV −0.0012、RB −0.0024；**BTE 关时 CNAS-on 在 5/5 均不优于 CNAS-off**（RT −0.0110 最明显）；**vanilla ≥ full 于 4/5**（单种子，待多种子坐实） |
| E-3 Patch | linksign | WikiVote@20000 + RedditBody@20000 | ✅ **完成（CN 修复版，2026-09-12）** | 8/8：RB AUC 0.9149–0.9284（无单调趋势）、WV 0.9594–0.9619（近持平）→ patch 不敏感、默认 P=1 有据；已同步并替换 `results/E-3_patch/`（汇总已更新） |
| E-4 时序 | linksign | WikiVote@20000 | ✅ **完成（CN 修复版，2026-09-12）** | TD 0.9565 vs TE 0.9607（ΔAUC −0.0042、ΔAP −0.0120、ΔsignF1 −0.0118）→ 同配下 TE 略优、默认 TE 保留；汇总 `results/E-4_time_decay/E4_time_decay_summary.md` |
| E-5 显著性 | sign + linksign | RedditTitle@20000 | ✅ **完成（CN 修复版，2026-09-12）** | linksign auc **0.9391±0.0007**（ap 0.7184±0.0004、sF1 0.8779±0.0077）；sign auc **0.6746±0.0028**（ap 0.9401±0.0019）；汇总已替换；p 值待基线同口径数据（基线轨见 HANDOFF 09-12） |
| 主表重跑 | sign + linksign | 5 数据集 | ✅ **完成（CN 修复版，2026-09-13）：50/50 runs** | **auc（5 种子）**：sign RT 0.6746±0.0028 / RB 0.6117±0.0354 / WV 0.7956±0.0020 / BA 0.7872±0.0086 / OTC 0.8696±0.0048；linksign RT 0.9391±0.0007 / RB 0.9235±0.0045 / WV 0.9612±0.0005 / BA 0.9574±0.0019 / OTC 0.9708±0.0025；raw + sha256 已归档（`results/main_tables/raw/`、`_sync_raw_log.csv`）；基线模型不受 CN 修复影响（Baseline 线独立） |
| E-6 异配图 | — | — | ⛔ 本轮不做 | — |

**执行顺序（固定，不跳步）**：E-3(GPU重跑) → E-2(GPU) → E-4 → E-5 → 主表重跑

**设备基座（2026-09-06）**：最终进论文表格的数据**统一 GPU**（同 seed 跨设备不可比，CPU/GPU 随机流不同）；CPU 期日志/结果仅作存档与冒烟参考，不进论文。结果 JSON 现含 `device` 字段可核验。

## E-3 结果摘要（2026-09-12，CN 修复版重跑，详见 results/E-3_patch/E3_patch_summary.md）
- RedditBody：AUC 区间 0.9149–0.9284（P7 略高、无单调趋势），Sign-F1 P1 最高 0.9328
- WikiVote：AUC 0.9594–0.9619 近持平（跨度 0.0025），不敏感
- 结论：patch size 不敏感，默认 patch_size=1 有据可依（回应 R2#8）；⚠️ 旧“P1→P7 单调下降（−0.021）”表述作废

## 审计发现（2026-09-09，subagent 只读核查 + 数据实测）

> **RAS 与 RAE 实际为（近）无效果模块** —— 影响 E-2 消融结论与论文消融表述，**需与导师/Agent A 定夺**（决策入口已发 HANDOFF）。

1. **RAS（repeat_aware）语义错位**：`utils/direct_neighbor_sampler.py` `common_neighbor_location`（:94-108）的 repeat 分支只**改写已有键**（要求 dst/src 出现在自己的历史邻居里 = 自环），从不把普通重复对插入锚点。实测：Bitcoin/Reddit **0 自环 → 0 触发**；WikiVote 5 条自环 → 14/20000 (0.07%) 边触发 → 与 E-2（4 数据集 base≡full、仅 WikiVote 微差）**定量吻合**。
2. **RAE（np.append 修复）仍是 no-op**：`models/NeighborInteractEncoder.py:293-297` 修复赋值真实存在，但 append 的 src/dst 或本就在交集中（padded 序列位置 0 = 自身），或凑不成配对 → on/off 输出**逐元素等价**（单测 identical=True）。parameter count 相同因 RAE 复用 BTE 的 `neighbor_sign_effect_layer`（共享层，无新增参数）。
3. **启动脚本参数链路正确**：run_experiments MODULE_GROUP→子进程 flag、args→sampler/model、`-r` 排除 / `-e` 5 种子均无误。
4. **结论**：E-2 的 base≡full 是**真实恒同**（非舍入巧合）。若论文消融声称 RAS/RAE 贡献，在 4/5 数据集不成立。
5. **连带影响**：主表「full 模型」实际等价 BTE+CNAS；**E-3/E-4/E-5（全模型）数据仅在 C5=方案②（不修 RAS/RAE，full:=BTE+CNAS）下为最终口径**；若 C5=方案①（修好 RAS/RAE），full 模型将改变 → E-3/E-4/E-5/主表**全部需重跑**（E-3 已被 A 写入论文 §4.5，需 A 知悉此条件性）。

## 🔴 追加重大发现（2026-09-09，subagent 深挖 BTE）：**pos0 标签泄漏**
- `models/SignDyGFormer.py:534`：padded 序列 pos0（自身 token）的 sign = **当前待预测边 (u,v) 的标签符号**；训练与**评估**都传真标签（`evaluate_models_utils.py:147` 等）。
- BTE 共同邻居计数中，重复边 (u,v) 会使配对 `(u@seq(u)pos0 × u@seq(v)历史)`、`(v@seq(u)历史 × v@seq(v)pos0)`，`suggest_sign = 标签 × 历史符号`（`NeighborInteractEncoder.py:169-181`）→ **测试时标签泄漏**（仅重复边，Bitcoin 重复率 ~40%）。
- **影响面**：所有 BTE 开启的运行（E-2 四行全部 BTE=True、E-3/E-4/E-5/主表、旧主表）→ 重复边上指标**虚高**；基线模型（无 BTE）不受影响 → **full 模型超基线的核心结论被泄漏系统性高估**。RAE no-op 依旧。
- **修复方向**（subagent 建议）：计数前剔除 pos0/自身 id（indirect 只收真第三方 w∉{u,v}）+ RAE 改为在 counterpart 位置按历史符号直写 [1,0]/[0,1]（不用标签）→ 修复后需全量重跑。
- **修复状态（2026-09-10 更正）**：代码修复已实现（改动 1-4，见 `docs/FIX_PLAN_RAS_RAE_LEAK.md`）；**本地验证全绿**（泄漏消除/RAE 增量/RAS 触发 31/300）。⚠️ **部署事故（09-09）**：服务器 `git pull` 在后台链中被中断（fetch 成功、快进未完成、静默失败且未校验）→ 服务器代码一直停留 `38322c6`；**此前所有"修复后"结果（含 09-09 验证 run 与 E-2 第 2 批）均为旧代码运行，结论全部作废**（同 seed 确定性 → 与旧结果逐位相同，仅为复现）。✅ 已重新正确部署：server HEAD=`cefb9ac`，补丁标记 + **服务器端采样差异 9/101 验证通过**。
- **修复后验证结果（09-10 实测 · 真实影响）**：① **BA linksign full：AUC 0.9649 → 0.9554（−0.0095）**、AP −0.036、sign-F1 −0.024；② **RB linksign full：AUC 0.9610 → 0.9299（−0.0311）**、AP 0.7879→0.6948（−0.093）、sign-F1 −0.040 → **泄漏曾系统性高估，影响重大；"无影响"结论彻底推翻**；③ E-2 RT 消融（修复后部分）：base 0.9337 / +RAE 0.9348 / +RAS 0.9366 → **RAS/RAE 出现小幅非零效应**（full 运行中）。
- **已完成（2026-09-11）**：E-2 修复后全量 **20/20**（双卡，最后完成 17:01）+ E-5 修复后 **10/10**（队列自动）。结果已同步到本地归档（`tools/sync/fetch_results.py`；sha256 见 `results/_sync_raw_log.csv`）；汇总表 `E2_ablation_summary.md`/`E5_summary.md`。
- **待办（影响面扩大）**：E-3/E-4/E-5/主表全量重跑；"SignDyG 仍优于基线"需以修复后主表重新验证；上报导师。

## E-2 修复后消融 全量结果（⚠️ 已作废 · 2026-09-11 17:01 · CN 修复前语义）

| 数据集 | base (RAS-,RAE-) | +RAE | +RAS | full | 旧代码（含泄漏，全配置同值） |
|---|---|---|---|---|---|
| BitcoinAlpha | 0.9524 | 0.9532 | **0.9606** | 0.9554 | 0.9649 |
| BitcoinOTC | 0.9676 | 0.9680 | **0.9746** | 0.9720 | — |
| WikiVote | 0.9620 | 0.9630 | 0.9628 | 0.9629 | — |
| RedditTitle | 0.9337 | 0.9348 | 0.9366 | **0.9373** | 0.9539 |
| RedditBody | 0.9265 | 0.9249 | 0.9282 | **0.9299** | 0.9610 |

**观察**：① 修复后全部低于旧值（**泄漏曾高估**：BA −0.010 / RT −0.017 / RB −0.031）；② **RAS 5/5 一致正增益**（+0.0008~+0.0082）；③ **RAE 单独≈中性**（−0.0016~+0.0011）；④ **full ≥ base（5/5）**，增量主要来自 RAS；Bitcoin 两数据集 full<+RAS（BA −0.005、OTC −0.003），RT/RB full 略优；⑤ 明细（AP/sign-F1/耗时）见 `results/E-2_ablation/E2_ablation_summary.md`，归档同步记录 `results/_sync_raw_log.csv`。

## 待决策（阻塞主表重跑与 E-2 定稿）
- **RAS/RAE 去留（C5）**：语义定义见 `docs/DESIGN_RAS_RAE_FIX.md`。**用户 09-09 定案：以论文定义为准**（git 溯源跳过）；已向 Agent A 请求摘录论文中 RAS/RAE 准确定义（HANDOFF）。Agent A 曾倾向方案②——其依据（"泄漏无实际影响"）已因部署事故作废；C5 定案待修复后重跑数据。主表重跑仍暂缓；E-4/E-5 汇总先行交付。
- 建议先与导师确认（含机制 A/B 与是否新增通道）。
- **CN 伪交集怪癖**：**已决策（用户 09-11）：✅ 修复**（真交集；RA 开/关均受影响已实证，与 RAS 无关，差异率 48–79%）。全量重跑已排程（队列 15 任务条）；**旧结果（含伪交集）全部作废，报告以重跑后为准**。分析+实证：`docs/ANALYSIS_CN_PSEUDO_INTERSECT.md`。

## 每次运行后需记录

对每个 run（或每个实验），记录：
- **开始/结束时间**、耗时
- **异常情况**（如有）
- **结果文件路径**（`saved_results/...`）与关键指标
- 更新上方"实验状态总览"表格

## 已确认决策（2026-09-05 更新，以 ADVISOR_DECISIONS.md 为准）

- 数据集设置**保持 tail 现状**：RedditTitle / RedditBody / WikiVote tail20000；BitcoinAlpha/BitcoinOTC 全量
- **E-2 消融 = 导师方案 4 组 × 全部 5 数据集**（CNAS+BTE 基座，RAS/RAE 解绑；link&sign）；配置已固化
- E-2 旧方案（基线全关→逐加）在 RedditTitle/WikiVote 已跑完，其中"全开"配置可复用，新跑用 `--module-idx 0 1 2` 跳过
- E-3/E-4 用 **link&sign** 任务；E-5 用 **sign + link&sign 双任务 × 5 种子**（满足审稿人 R2#8）
- **RAE bug 已修复**（`np.append` 未赋值）：旧主表（sign-ms.csv/linksign_ms.csv）用旧代码跑出，须用修复后代码重跑；先 BitcoinAlpha 影响评估（阈值 0.5%）再决定全量
- E-6（异配图）、E-7（噪声）：本轮不做（回复中作 future work）
- SEMBA 在独立仓库，不在此实现

## 运行日志与结果位置

| 内容 | 路径 |
|---|---|
| run_experiments 批量日志 | `expm-YYYY-MM-DD-logs/{任务}/` |
| 结果 JSON（含 E-1 效率 4 项） | `saved_results/{LinkSign\|SignLinkPrediction}/{model}/{dataset}/` |
| profiler 推理明细 | 同目录 `{...}-profiler.json` |
| 统计汇总（mean±std + p 值） | `dataset_analysis/compute_stats.py` 输出 |

## 最近更新记录

- **2026-09-14 午间（续8·波二 env 构建启动）**：用户认可波二方案：**批准建 `dygmamba` / `scadyg` 两个 conda env**；smoke 编排在 #61/#62 之后；**结果整理后交 `Paper`**。已落地：新增 `tools/server_setup/build_env_dygmamba.sh`（torch2.1+cu118 → 预编译 causal-conv1d/mamba-ssm → 失败回退 conda cuda-nvcc=11.8 源码构建 → CUDA 算子自检）与 `build_env_scadyg.sh`（旧栈：torch1.12.1+cu116 / PyG 轮子 / dgl1.0.0 / deepsnap / py-tgb；含镜像回退）——经 git 通道部署后于服务器 **nohup 后台构建**（日志 `~/envbuild_{dygmamba,scadyg}.log`）；env 自检通过后追加 smoke 行。

- **2026-09-14 午间（续7·#61/#62 复核入队 + R1 波二基线就绪）**：① 用户批准「两个候选点都跑」：队列追加 **#61 WV 20/20 ×5 种子**（12:17 GPU0 开跑）与 **#62 OTC 40/15 ×5 种子**（GPU1 接续）；出数后同步（nh5b）。② **R1-4 波二基线：`Baseline` 就绪应答**（`mb-20260914-121230-baseline-de27`）：**DyG-Mamba GO(P1)**（基于 DyGLib 零数据转换；适配 v1+幂等守卫+--force 就绪；硬依赖 causal-conv1d/mamba-ssm 需编译）、**ScaDyG 条件 GO(P2)**（torch1.12.1+cu116 旧栈，必须独立 env）、**UniDyG/多模态文本 → 引用+讨论**（无开源仓库 / 数据层 barrier；符合 R1-4 官方口径）。③ **Code 预备**：Baseline 本地仓库（HEAD `1100ee6`）已推裸库 → 服务器 clone `pull --ff-only` 至 `1100ee6` ✓；**服务器核查：无系统 nvcc**（拟在 env 内装 conda 版 cuda-nvcc，免 sudo）、磁盘 116G 充裕；smoke 命令行已 staged（Phase A：BA×seed42×两模型→隔离 outputs/_smoke；Phase B 全量待 smoke 确认）。**⏸ 唯一硬门槛：待用户批准建 dygmamba / scadyg 两个 conda env。** ④ 另收 `Perf` **K3 交付**（用户已批；Rust 内核 vs numpy 2.94–3.95× 逐位一致）——Code 侧待办：accel ABI 升位 `k1k2k3-v2-2026-09-14` + wheel 重建 + 接缝替换 + 2× 路径核对 + golden 用例（待排期）。

- **2026-09-14 白天（续6·VPN 断线后全量同步：补跑完成 + Baseline GPU 完成）**：凌晨 03:40–12:1x 用户侧 VPN 断线（服务器持续在跑）；恢复后对账：① **补跑 #55–#60 全成功**（OTC 40/15=0.8835、80/5=0.8556、80/15=0.8631；WV 20/10=0.7963、60/10=0.7930、60/20=0.7954；无 Traceback；新代码下单 run 快约 2×）；② **Baseline GPU 完成**：DySDGNN BA 0.7511±0.0427 / OTC 0.9119±0.0122 / WV 0.5937±0.0656；DynamiSE BA 0.5360±0.0070 / OTC 0.4850±0.0016 / **WV 0.5714±0.1094（GPU 方差远大于 CPU 版 0.5667±0.0368）**——与 CPU 口径几乎一致（除 WV-DynamiSE）→ 已跑 `m5_summary`；③ **同步**：`--set nh5`（22 文件）+ 新增 `--set base-gpu`（30 JSON + 2 CSV → `results/baseline_m5/`；sha256 入 `_sync_raw_log.csv`）；④ **OTC/WV 完整画像**：均无点过 ≥+0.01 阈值；**WV 20/20=0.8002（+0.0065，≈3σ @WV 噪声±0.0020）→ 建议 1 次 5 种子复核（待批准）**；OTC 40/15=0.8835（+0.0086，≈1.8σ）边缘可选；⑤ 结论已更新（`SignNeighborhood_summary.md`）；Baseline 产物交付通知已发。

- **2026-09-14 凌晨（续5·网格策略认可 + 第二波基线问询）**：用户认可「网格扩展到此为止」：不再加密网格点；#55–#60 完成后仅当某 OTC/WV 点单种子显著超现行（≥+0.01 且邻点呈趋势）才对该点做 5 种子复核；RT/RB/BA 封盘（依据：响应面平坦 + 种子噪声主导 + 统计功效计算）。已向 `Baseline` 发**第二波外部基线问询**（`mb-20260914-033750-code-29f6`：DyG-Mamba / ScaDyG / UniDyG / 多模态文本的可行性评估、上队依赖与时间线；其仓库已有 R1-4 接入包 + 适配 v1 提交）。**队列实况**：#53 DySDGNN GPU（GPU0 运行中）、#54 DynamiSE GPU（GPU1 运行中）；#55–#60（sign 补跑）等待接续；双卡预计 ~2–3h 后空闲。

- **2026-09-14 凌晨（续4·rmtree 修复部署 + 补跑入队）**：① **修复**：5 个训练脚本（sign / linksign / direct / direct-sign / 3class）由整目录 `shutil.rmtree` 改为**仅清理本 run 目标文件**（`.pkl` / `.param.json` / `_nonparametric_data.pkl`；按各自 EarlyStopping 命名变量取文件），串行语义与"同名旧文件防误载"不变；移除 5 处 `shutil` import；本机 py_compile 全过（提交 `9956dd7`）。② **部署**：服务器 `pull --ff-only` → `9956dd7`（含 CN numpy 向量化 + 本次修复）；服务器侧 `test_cn_vec_seam.py` **ALL PASS**；标记核对（`_stale_name` / `cn_counts_vec`）通过。③ **补跑入队**：追加 **6 条任务 #55–#60**（OTC 40/15·80/5·80/15、WV 20/10·60/10·60/20，按原命令；修复后同数据集并发已安全，无需串行循环）；待 Baseline #53/#54 跑完后自动接续。④ **Baseline**：#52 `ARCHIVED_CPU`（旧产物已归档 `outputs.cpu.bak-20260914`）；#53 DySDGNN（GPU0）/ #54 DynamiSE（GPU1）GPU 运行中。

- **2026-09-14 凌晨（续3·Baseline 上 GPU + 互删缺陷影响分析）**：用户指示 Baseline 上 GPU：队列追加 **#52 归档（`outputs`→`outputs.cpu.bak-20260914`，旧 CPU 产物保留）+ #53 DySDGNN GPU 15 runs + #54 DynamiSE GPU 15 runs**（此前 #49/#50 因 `m5_run` 幂等全 skip——clone 内已有其本地 M5 CPU 产物且无 `--force`）；完成后 Code 手动跑 `m5_summary` 并同步；已发 `Baseline` 通告。**互删缺陷影响评估（回应"best 收尾"疑问）**：收尾流程 = early-stop 后 `load_checkpoint`(best) → 测试集评估 → 写 JSON；因此 best 缺失⇒训完前崩溃（响亮，6 点的表格中就是缺值），**不是静默错值**；已成功的 16 个结果无隐性污染（成功=结尾加载的必是自有 best；且胜者 seed42 重跑与网格 seed42 指标逐一相同 = 决定性复核）；残余风险仅"同名文件"（完全同 config 并发，本批未发生）——修法仍建议改为仅清理本 run 目标文件（待批准）。**5 种子初步结论**：RT Δ+0.0078（t≈1.3）、RB Δ+0.0213（t≈0.95）均 n.s. → **主表不动**；RB 候选方差更小（±0.0185 vs ±0.0354），可作敏感注脚。

- **2026-09-14 凌晨（补充批战报 + 结果整理，用户指示同步）**：队列 **#34–#51 全部派发完毕**（running.txt=51/51）；#34/#35 胜者 5 种子、RT 探边 2 点、OTC/WV 邻域 4 点已出并同步（`--set nh5`，16 文件 + sha256 入 `_sync_raw_log.csv`）。**关键结论（写入 `results/sign_neighborhood/SignNeighborhood_summary.md`）**：① RT 胜者 NN-100/LF-3 5 种子 auc **0.6824±0.0124** vs 现行 0.6746±0.0028，**Δ=+0.0078（t≈1.3，n.s.）**——单种子 +0.0300 系 seed42 特异；② RB 胜者 NN-40/LF-1 **0.6330±0.0185** vs 现行 0.6117±0.0354，**Δ=+0.0213（t≈0.95，n.s.）**——主源 seed123（现行低值 0.5475）；③ 均**不更新主表**（保持 NN-100/LF-1 / NN-60/LF-1）；④ OTC/WV 已出 4 点均在噪声带内。**⛔ 发现并发互删缺陷**：`train_link_sign_prediction.py` L266 开局 `rmtree(同数据集同 seed 模型目录)`，同数据集双卡并行时互删 best `.pkl` → 结尾 `load_checkpoint` FileNotFound，**6 点失败**（#37/38/39 OTC、#41/43/44 WV；日志 task_37/38/39/41/43/44）；补跑方案 = 两条串行循环 @ 任务（待批准）；修法（待批准）：rmtree 改为仅清理本 run 目标文件。**Baseline #48–#51 已执行**：#48 env `[OK]`（WARN 系快退出误报）、#49/#50 `m5_run` **全 skip**（clone 内已含本地 M5 CPU 版 outputs；脚本无 `--force`）→ 当前 summarize 为 **CPU 口径**（BA 0.7507 / OTC 0.9119 / WV 0.5935 DySDGNN；DynamiSE 0.5360/0.4850/0.5667）；是否强制 GPU 重跑待议（拟问 `Baseline`）。**部署 Hold 仍有效**：待 6 点补跑完再 pull（保持同代码口径）。
- **2026-09-14 凌晨（续2·CN 重构评估，用户要求）**：新增 `tools/verify/cn_bincount_bench.py`——**bincount 直方图重构 vs 现行 numpy 版**：**不如现行**（BA 口径 1.29ms→3.24ms＝**0.40×**；全配置中位 0.42×，仅 B=18 尾批 1.73×；逐位一致全过；大 id 另有**域膨胀爆内存**缺陷：直方图域=B·n_nodes）。含 AVX 推论：CN 整 run 占比已 ~0.4–0.5%，真 SIMD 再快 3–5× 也仅再省 ~0.3% ⇒ **CN 加速以现行 numpy 版收官，不再投入 bincount/AVX/SIMD 子路线**；**用户认可**，K3-CN 请求**结案不立项**（已同步 `Perf`：`mb-20260914-005532-code-0631`；Rust 线留待其它热点）。
- **2026-09-14 凌晨（续·CN numpy 向量化落地，用户批准）**：用户拍板「**受 accel 参数控制**」落地 CN 过渡加速：`utils/accel.py` 新增 **`cn_counts_vec()`**（全批次向量化：行复合键全局 unique + searchsorted 跨行匹配；消除 2B 次 `np.unique` / 2B 次逐元素 `apply_` / 4B 次行级 `.to(device)`），`NeighborInteractEncoder.count_nodes_appearances` 顶部加**与 K1/K2 同款接缝**（默认启用；`--no-accel` / `SIGNDYG_ACCEL=0` 回原路径；K3 Rust 到货后同接缝替换）。**本地自检 `tools/verify/test_cn_vec_seam.py` 全绿**：逐位一致 10 场景（含 Ls≠Ld、单侧空、大 id、B=0 安全增强）+ 开关路由 spy + 契约；过程中抓到并修复一处边界语义（单侧空行时对侧内部计数不得被清零——与逐行原语义对齐）；`test_accel_seam.py` 回归仍全绿（K1/K2 真实内核逐位一致）。**部署 Hold**：服务器 pull 推迟到 **#34–#47 批次跑完**（保持该批同代码口径），随后 pull + 服务器侧自检（`test_cn_vec_seam.py`）＋可选短跑 OFF/ON 复核；#48–#51（Baseline）将自然使用新代码。K3 立项仍待用户排期（Perf 初评正面）。
- **2026-09-14 凌晨（补充编排·第 2 批，用户批准）**：队列追加 **#34–#47**（14 条）：**胜者 5 种子** RT NN-100/LF-3 与 RB NN-40/LF-1（各 1 条 × `--seeds 42 123 456 789 1024`）；**OTC 邻域 5 点**（40,5）(40,15)(80,5)(80,15)(60,15) 与 **WV 邻域 5 点**（20,10)(20,20)(60,10)(60,20)(40,20)（单种子）；**RT 探边 2 点**（100,5)(120,3)（单种子）。出数后按惯例同步；胜者通过则更新主表 sign RT/RB（5 种子口径）。另：已向 `Baseline` 发**编排问询**（`mb-20260914-000726-code-5d26`：就绪度/实验清单/队列格式/依赖需求）。**00:09–00:10 #34/#35 已自动派发双卡开跑；#36 起排队等待（双卡占用）。** **同步更新（00:15）**：Baseline **就绪回执已收**（HEAD `3b6f33518737cb4a526164c2667707aeec543f69`，含哈希更正 FYI）；**gc 依赖核查：pydantic 2.12.5 / PyYAML 6.0.1 / scikit-learn 1.4.2 均在 → 零安装**（无需 pip 改动）；部署（git 通道：裸库+clone 替换旧 tar 拷贝）与队列 **A→B/C→D** 待用户许可后执行。`Perf` 对 K3 请求**初评正面**（正式立项待用户确认排期）。**00:2x 部署执行完成（用户批准）**：git 通道（新建裸库 `~/git/DynamiSE_DySDGNN.git`，临时 mirror push）→ 旧目录移至 **`~/wyq-exprm/DynamiSE_DySDGNN_repro.tarbak-20260914`**（备份不删）→ 服务器 clone HEAD=**`e6178cc`**（=回执 `3b6f335` 的下一提交：信箱同步+check_deps+NOTES）；`check_deps.py` → **[OK] deps ok**（torch 2.2.2 / CUDA True×2 / pydantic 2.12.5 / pandas 2.3.2 / sklearn 1.4.2，零安装）；队列追加 **#48–#51**（Baseline A 自检→B DySDGNN（P1 15 runs）→C DynamiSE（P2 15 runs）→D 汇总；幂等）。

- **2026-09-13 深夜（NVML 修复）**：用户重启服务器完成驱动对齐（运行中 595.84 vs 用户态 595.91 错配；DKMS 595.91.07 早前已编译好，重启即生效，无需重装）。验证：`nvidia-smi` 正常（Driver 595.91.07 / CUDA 13.2 / 2× RTX 2080 SUPER 空闲，仅 Xorg 4MiB）；`/sys/module/nvidia/version`=595.91.07；`gc` 环境 torch CUDA=True×2；**队列守护已重新拉起**（`running.txt` 保留 33/33，不会重跑旧任务；`/tmp` 锁已随重启清空）；队列 idle，**可接新任务**。记录见 `docs/GPU_DRIVER_INSTALL.md` 文末。
- **2026-09-13（晚·续）**：**ⓑ BTE-off 10/10 + ⓓ 邻域网格 15/15 全部完成并同步**（`--set e2b/e2d`，sha256 已入 `results/_sync_raw_log.csv`；队列 33/33 于 19:44 全部 idle）。**ⓑ（单种子 42）：BTE 边际效应 = BTE开−BTE关（同 CNAS-E）**：BA −0.0002（≈0）、OTC +0.0053、RT +0.0081、WV −0.0012、RB −0.0024；**BTE 关时 CNAS 效果**（CNAS-E−CNAS-D）：**5/5 均为负**（BA −0.0061、OTC −0.0040、WV −0.0050、RT −0.0110、RB −0.0018）；**全关 vanilla vs 全开 full**：vanilla ≥ full 于 4/5（BA +0.0031、OTC +0.0003、WV +0.0063、RB +0.0095；仅 RT full +0.0046）——单种子口径，若进论文需多种子。**ⓓ（sign 任务 NN/LF 网格，单种子）**：RT 网格最优 **NN-100/LF-3 = 0.7070**（其余点 0.6589–0.6761）、RB 最优 NN-40/LF-1 = 0.6485、BA 最优 NN-40/LF-10 = 0.7813；是否用赢家加 5 种子复核待定。
- **2026-09-13（晚）**：**CN 共现编码细粒度评估完成（K3 前置；本机 CPU 全跑，无需服务器）**：新增 `tools/verify/cn_microbench.py`（逐位复刻 `count_nodes_appearances` + 12 子步骤分解 + 全向量化原型对照 + 逐位一致校验）。要点：① 本机 BA 口径（B=200, L≈40）原实现 **≈37–41ms/次**，**逐行固定成本占 67–81%（其中 np.unique×2 = 37–42%）**、逐元素 `apply_` 14–29%；**向量化原型 28× @L40（14–55× @L16–100），逐位一致**；② 服务器测试阶段 profiler 对账：CN ≈130–141ms/次，**不受 CNAS/BTE 等模块开关影响**（vanilla 全关仍 137.7ms）⇒ 无条件计算；每次测试轮 ≈70 次 CN 执行 ≈9.6s ≈ 测试轮墙钟 **~70%**（轮墙 12.4–13.2s；min=13ms 对应 18 条尾批，核对通过）；单 run 测试段 CN 合计 42–80s；③ 训练段：BA 训练 64 batch/epoch（12,673/200），CN 估计 ~200–260s/run ⇒ **CN 合计约 10–13% run 墙钟**（无训练期 profiler，按同实现同批次估计）；④ 本机限制：`.to(cuda)` 行级 H2D（4×B=800 次/调用）不可测，待服务器小样本标定；⑤ **队列 33/33 全部完成（19:44 idle）**：ⓑ BTE-off 与 ⓓ 邻域网格已跑完，随本轮执行全量同步。产物：`tools/verify/cn_microbench_result.json`、`cn_microbench_sweep.json`。**K3 提案已发 `Perf`**（`mb-20260914-000310-code-7f4d`，用户批准；含热点证据/子步骤分解/向量化原型/建议接缝与门禁）。
- **2026-09-13（傍晚）**：**补充实验进展**：ⓐ **P 补全 12/12 完成并同步**（BA/OTC/RT × P{1,3,5,7}）；全数据集 P 表引入 `results/E-3_patch/E3_patch_summary.md`（R2-11 数据齐）。ⓑ BTE-off 对照 **5/10**（BA/OTC 完成、RT 1/2 在跑；预计 ~21:30 齐）；ⓓ 邻域网格 **7/15**（RT 5/5 完成、RB 2/5；BA 待跑；单 run 10–25 分钟，预计 ~22:00 齐）。拉取集已预备（`--set e2b / e2d`），齐后一键同步。
- **2026-09-13（下午）**：**E-2 v3（CN 真交集修复版）20/20 完成并同步**：v1/v2 作废、**「RAS 5/5 正增益」「full≥base 5/5」表述作废**（伪交集产物）；v3（单种子 42）：full 在 RT（+0.0075 ≈10σ）与 BA（+0.0032）为正、其余 ≈ 噪声带；RAE 单独中性偏负（RB −0.0172）；RT/RB full 与 E-5/主表 seed42 逐字一致（0.9386/0.9239）。**BTE-off 对照（ⓑ）今晚跑（队列 #18）**。汇总 `results/E-2_ablation/E2_ablation_summary.md`；raw 20 个 sha256 入 `_sync_raw_log.csv`（14:27 批次）。
- **2026-09-13**：**统一信箱（独立仓）切换完成（M6，用户批准，Code 执行）**：正式通道 = `D:\codes\agent-mailbox`（JSONL 真源 + CLI/MCP 工具；59 项测试全绿）；旧箱 **40 条全部导入**（code 23 / paper 3 / perf 14 / baseline 0；对账 + 抽样逐字全过，清单 `data/legacy_import_manifest.json`）；`docs/handoff/*` **冻结只读**；切换公告已发三箱（新箱 `mb-20260913-095657-code-5655` / `…-98fb` / `…-148c`，待回执）；M7 = 各 agent 自行自同步；**3 天双通道对账至 09-16**。
- **2026-09-12（深夜3）**：**服务器构建校验「第 1 批」全绿**（按 Perf 清单执行）：cargo test **21/21**、check_accel **12/12**（bit-exact）、check_ref **8/8**、pytest **21 passed**、上游对照真实 **K1 1800/1800 + K2×2** 与合成 **908 查询**全 PASS、bench_real **14/14**；**服务器口径提速**（Linux/40 核）：K1 7–11×、K2 50–285×（L15 267–285× → L100 50–53×）。唯一遗留：crate 侧 `__abi__` 待 Perf 补（M4 前置）→ **已闭环**：Perf 补契约串（`374beb5`）→ 服务器重建复验全绿（**新 wheel sha `1ffe3216…`；门 1/2 闭环**）——第 1 批含 ⑥ 全部完成。
- **2026-09-12（深夜2）**：**Perf 服务器部署就绪**（用户批准执行）：独立仓通道 `~/git/SignDyG-Perf.git` + `~/SignDyG-Perf` @ `9b4913b`（Code URL 直推；Perf 零接触）；Rust 环境（rustup/cargo 1.98.1、maturin 1.15.0；用户态 + TUNA）；**首构建 25s 通过**（wheel `signdyg_accel-0.1.0-cp39-cp39-manylinux_2_34`，sha256 `86c2e85f…`；gc 安装 + import 校验 ✓）。剩余：crate `__abi__` + DyGLib 适配器（M4 落地，待批准）→ 门 2/3。
- **2026-09-12（深夜）**：**信箱拆分（一人一箱）**：`docs/HANDOFF.md` 单文件 → 入口页 + `docs/handoff/outbox-{paper,code,baseline,perf}.md`（唯一写入者=箱主；收件方在自己箱内回执）——消除单文件多写者旧缓冲覆盖（本日曾 2 次）。各 Agent 工作区契约指针已同步更新。
- **2026-09-12（晚）**：**主表第二批同步完成**（linksign RT+RB × 5 种子，sha256 已记录）→ **RT、RB 双数据集主表全部完成**（linksign RB auc **0.9235±0.0045**、ap 0.6884±0.0043、sF1 0.9361±0.0039；RT 0.9391±0.0007；RB seed42=0.9239 与 E-3 P1 独立复跑一致 ✓）。队列：WV sign ✓、BA sign ✓；WV linksign 3/5、BA linksign 2/5、OTC 排队。⚠️ NVML 仍待重启修复（建议本轮队列跑完后安排）。
- **2026-09-12（午）**：**E-5 10/10 完成并同步**（linksign auc 0.9391±0.0007、sign auc 0.6746±0.0028；`E5_summary.md` 已更新；同时构成主表 RedditTitle 行）。**基线轨启动**：拟把 DynamiSE/DySDGNN（Baseline 复现）及后续方法编排进连续队列——已向 `Baseline` 发**接入汇报需求**（HANDOFF 09-12）；已核实前置：服务器旧拷贝为 09-09 tar+scp（无 git，需 git 化重建）、`gc` 环境缺 `torchdiffeq`（待用户许可后安装）；暂存任务行 `tools/queue/tasks.baseline.staged.txt`。
- **2026-09-12（上午）**：**E-3/E-4 CN 修复版重跑完成并同步归档**（E-3 8/8、E-4 1/1，sha256 见 `results/_sync_raw_log.csv`；汇总已替换）；**主表 sign RT/RB 5 种子完成同步**（auc RT 0.6746±0.0028 / RB 0.6117±0.0354）；linksign RT/RB 运行中。⚠️ **服务器 NVML 故障**：无人值守升级（06:35，新内核 6.8.0-138 + 驱动用户态 595.91）导致 `nvidia-smi` 报 driver/library mismatch（内核模块 595.84 未重载）；**CUDA 训练不受影响**（新进程 `is_available()=True`、任务正常）；建议本轮队列完成后安排重启修复（待用户定）。
- **2026-09-11（晚）**：用户决策**修复 CN 伪交集**（assume_unique→真交集）；受影响实验全量重排（队列 15 条，含 @ 原生命令支持）；旧结果作废，报告以重跑后为准。**22:20 修复版已部署至服务器（pull→`9d658a9`），本地/服务器双层语义校验通过（keys=[2]）；新队列守护进程 pid=239314 启动，task#1 E-4 TD（GPU0 pid=239335）与 task#2 E-3 RB（GPU1 pid=239435）已上卡。**
- **2026-09-11**：E-2 20/20 + E-5 10/10 完成（**伪交集旧语义，已作废**，重跑排程序列中）；归档同步本地（sha256 记录）；汇总表落盘（数字已被重跑取代）；Perf M0–M2 完成（加速内核双闸门初步达成）。
- **2026-09-01**：服务器恢复；代码推送至服务器；`server_setup.sh`（数据就绪脚本）+ 本进度文件建立；E-7 确定本轮不执行。
- **2026-08-20**：E-1~E-7 全部代码实现完成（时间衰减、效率测量、消融/patch 脚本、统计脚本、噪声模块、RAE bug 修复），本地 CPU 冒烟测试全部通过；`EXPERIMENT_PLAN.md` 运行计划定稿。
