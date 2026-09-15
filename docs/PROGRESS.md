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

## 术语约定（2026-09-15 用户拍板）
- **base ≡ w/o all ≡ [F,F,F,F]**（全模块关闭；历史别名 `vanilla`、历史行号 idx 5）——此后文档/汇报/邮件统一用 **base**。
- **[F,F,T,T]（BTE+CNAS）**：历史文档/代码曾称 "base"/"基座"（导师方案 R1-5 用语）→ **即日起统一称 `BTE+CNAS`，不再叫 base**（行号 idx 0 不变，`--module-idx 0`）。
- 其余：`full`=[T,T,T,T]；`CNAS-only`=[F,F,F,T]；`BTE-only`=[F,F,T,F]。
- ⚠️ 2026-09-15 前的旧记录中 "base/基座" 一律按 **BTE+CNAS** 理解（含 #81 任务）；新记录按本约定。

## 实验状态总览

| 实验 | 任务 | 数据集 | 状态 | 结果路径 / 备注 |
|---|---|---|---|---|
| E-1 效率 | sign | RedditTitle@20000 | ✅ 可提取 | E-5 sign seed42 已含 4 项效率数据，待汇总 |
| E-2 消融 | linksign | **全部 5 数据集** | ✅ **完成（v3 CN 真交集，09-13）**：v3 20/20 + ⓑ BTE-off 10/10 + ⓓ 网格 15/15 均已同步 | v3 汇总 `E2_ablation_summary.md`；ⓑ（单种子）：BTE 边际 BA≈0、OTC +0.0053、RT +0.0081，WV −0.0012、RB −0.0024；**BTE 关时 CNAS-on 在 5/5 均不优于 CNAS-off**（RT −0.0110 最明显）；**vanilla ≥ full 于 4/5**（单种子，待多种子坐实） |
| E-3 Patch | linksign | WikiVote@20000 + RedditBody@20000 | ✅ **完成（CN 修复版，2026-09-12）** | 8/8：RB AUC 0.9149–0.9284（无单调趋势）、WV 0.9594–0.9619（近持平）→ patch 不敏感、默认 P=1 有据；已同步并替换 `results/E-3_patch/`（汇总已更新） |
| E-4 时序 | linksign | WikiVote@20000 | ✅ **完成（CN 修复版，2026-09-12）** | TD 0.9565 vs TE 0.9607（ΔAUC −0.0042、ΔAP −0.0120、ΔsignF1 −0.0118）→ 同配下 TE 略优、默认 TE 保留；汇总 `results/E-4_time_decay/E4_time_decay_summary.md` |
| E-5 显著性 | sign + linksign | RedditTitle@20000 | ✅ **完成（CN 修复版，2026-09-12）** | linksign auc **0.9391±0.0007**（ap 0.7184±0.0004、sF1 0.8779±0.0077）；sign auc **0.6746±0.0028**（ap 0.9401±0.0019）；汇总已替换；p 值待基线同口径数据（基线轨见 HANDOFF 09-12） |
| 主表重跑 | sign + linksign | 5 数据集 | ✅ **完成（CN 修复版，2026-09-13）：50/50 runs** | **auc（5 种子）**：sign RT 0.6746±0.0028 / RB 0.6117±0.0354 / WV 0.7956±0.0020 / BA 0.7872±0.0086 / **OTC 0.8775±0.0054（09-14 换装 NN-40/LF-15；旧 NN-60/LF-10 = 0.8696±0.0048）**；linksign RT 0.9391±0.0007 / RB 0.9235±0.0045 / WV 0.9612±0.0005 / BA 0.9574±0.0019 / OTC 0.9708±0.0025；raw + sha256 已归档（`results/main_tables/raw/`、`_sync_raw_log.csv`）；基线模型不受 CN 修复影响（Baseline 线独立） |
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

- **2026-09-15 晚（续35·「C/R 锚点需要不同 look-back」量化 + 叙事方案）**：用户直觉：C 与 R 锚点需不同窗口半径。代码核对：当前实现与论文 §3.2 均为**单一 k 共用**。扩展 `cnas_last_cn_stats.py`（新增 Rshare / covR(k=0) 列）实测：**R 窗对并集贡献** RB **36.3%** / RT 27.1% / OTC 22.7% / BA 6.2% / WV 0.1%；**R 收窄到 k_R=0 的覆盖率损失** RB −7.8pp / OTC −6.3pp / RT −3.2pp → 两锚点角色不同、单 k 存在耦合。**论文叙事（Code 建议）**：⚠️ 不可写成"原实现未用对参数/我们修 bug"（§3.2 现行即是单一 k、实现与之一致）；应采**角色分解（decoupled role-aware look-back）**框架 = C 窗（证据上下文）与 R 窗（直接交互 + recency 回补）解耦为 k_c/k_r，**且待 dual-k 实验出现性能增益后**再写入论文。**待批**：`k_R` 接口（默认=LF、零行为变更）+ 单种子网格 k_R∈{1,3,5,10} × RT/RB/OTC（≈12 runs）。

- **2026-09-15 晚（续34·「RAS+CNAS 联合加密」筛查实验入队）**：用户指示：安排 RAS+CNAS 联合加密实验，判断「**RAS 若能补回 CNAS 的 recency 损失 → CNAS 设计依旧可行**」。**设计（单种子 42 筛查）**：在密度扫描加密点上加 RAS（=+RAS [T,F,T,T] 版），与已收 RAS-off 同点对照配对——**RT@120/10、RB@160/10、BA@40/20、BA@80/20 共 4 runs**（RB 行含 `expandable_segments` 防 OOM）。**已插入队列 #86–#89**（`edit_remote_tasks` 插于 #85 后；k×N 网格顺延 **#90/#91**；队列 **91 行**）。**判读口径**：+RAS@加密点 ≥/≈ BTE-only（CNAS-off）且 ≥ BTE+CNAS（RAS-off）→ RAS 回补成立、设计维持；若 +RAS ≈ BTE+CNAS（无效果）或仍低于 BTE-only → 记 limitation/再议。正向则扩 5 种子。**现有反证提示**：full（含 RAS+RAE）在当前密度下 RT 的 ap 仍显著低于 BTE-only（−0.0073/t=−3.0）——本次即测「加密设定下是否反转」。

- **2026-09-15 晚（续33·CNAS「最新截断」：论文定义核对 + RAS 锚点回补量化）**：用户质疑「最新事件反而被丢」。① 核对 `docs/PAPER_RAS_RAE_SPEC.md` §3.2：**序列 = 锚点窗口 Concat → PadTop(N)，「止于最后锚点」是论文定义的直接推论（非实现走样）**；论文锚点 = C ∪ R（RAS 开时含 u-v 直接交互位置）。② 补算两口径（`cnas_last_cn_stats.py --repeat-aware`）：**锚点纯 C**（2×2 各配置）：平均丢最新 ~15–18%（BA 15.4% / OTC 15.0% / WV 16.0% / RT 18.3% / RB 15.6%）；**锚点 C∪R**（主模型 full 口径）：**OTC 15.0%→8.4%（中位 3→0）、RB 15.6%→12.0%、BA 15.4%→13.6%、RT 18.3%→16.7%、WV 不变**（重复对少）→ **RAS 恰好回补最严重的重复对端**（与 RAS 修复后小幅非零效应方向一致）。③ **待用户定**：①维持现状（论文自洽、limitation 如实写）；②改 §3.2（PadTop 前追加"最近 m 条"或 query 对自身作锚点）→ 改论文+代码+全量重跑（会报废当前队列中 S1/网格等部分算力，需尽快定）；③先做单数据集单种子 A/B 小验证再定。

- **2026-09-15 晚（续32·CNAS 加密机制对账 + last-CN 结构统计）**：回应「CNAS 加密理论上应退化到 recent-N」疑问：① **机制核对**（`utils/direct_neighbor_sampler.py` `look_forward_sampling`:787 + 并集注入:518）：窗口=[max(0,idx−k), idx] 且**被上一个 CN 截断**；序列=窗口并集（窗外事件全部丢弃）→ **k≥CN 间距时并集≈[起点, last-CN] 连续史 → 趋近 recent-N 成立**（与 RT@NN-120 上 on≈off 的观察相符）。② **新增离线统计 `tools/verify/cnas_last_cn_stats.py`**（CPU、每数据集 2000 边、测试段）：**last-CN 截断普遍存在**——平均 **~15% 的最新事件被截**（median 1–4 条；mean 8–21 条；p90 19–53 条），**且不随 k 增大而消失**（结构性）；即使 k=20，窗口并集覆盖率也只有 **0.67–0.79**（RT@LF1 仅 0.31）→「退化」是渐近且有界的；**no-CN 率**：RT 45.6% / RB 51.2%（≈一半边 CNAS 根本不生效）vs WV 14.8%。③ **判定**：BA 反常（80/20 −0.0106）**不能**归因于截断（BA drop% 15.4% 与其他数据集相当）→ 更可能为单种子波动；如需定论可补 2–3 颗种子。

- **2026-09-15 晚（续31·术语 + 结果已同步 Paper）**：已发 `Paper`（`mb-20260915-183732-code-4fef`）：① 术语规定（base≡w/o all；[F,F,T,T]→BTE+CNAS；旧记录解读规则）；② **BTE-only 5 种子结果**（5/5 胜 CNAS-only；WV 显著超 full；RT ap 显著负；待 #77 补 w/o all 后出 2×2 定稿）；③ **密度扫描 8/9**（收益≤0.006、过度加密反降；「密度不足非主因」；RB-160/10 补跑中）。

- **2026-09-15 晚（续30·术语规定：base＝w/o all）**：用户拍板：**base ≡ w/o all ≡ [F,F,F,F]**（历史别名 vanilla）；历史称 "base/基座" 的 **[F,F,T,T]（CNAS+BTE）即日起统一称 `BTE+CNAS`**（行号 idx 0 不变；#81 在跑的那条就是它）。已对齐：PROGRESS 顶部新增「术语约定」节（含旧记录解读规则）；`run_experiments.py`、`tools/verify/agg_configs.py`、`tools/sync/fetch_results.py` 的注释同步更新（纯注释/文档，零行为变更）。

- **2026-09-15 晚（续29·S1 修复重排 + 密度扫描结果 8/9 + OOM 补跑）**：① **S1 修复已部署**：`models/DyGFormer.py` 增加 `self.profiler = Profiler(); self.profiler.disable()`（镜像 SignDyGFormer；空记录 summary/save 安全）→ 提交 `17632bd`，服务器 `pull --ff-only` 校验（HEAD=17632bd、grep profiler=5 命中）；两条 S1 任务经新工具 `tools/queue/edit_remote_tasks.py`（ssh+python+base64，零转义；自动备份+回读校验）插入队列 **#83/#84**。② **密度扫描 8/9 已取回**（`--set dens` → `results/E-2_ablation/raw_density/`；单种子 42；base=CNAS-on / BTE-only=CNAS-off）：**收益普遍 ≤0.006、噪声量级**——RT base 0.9311(@60/1) → 60/10 **+0.0001**、120/10 **+0.0025**；RB base 0.9292(@80/3) → 80/10 **+0.0031**；BA base 0.9554(@40/15) → 40/20 **+0.0045** 但 80/20 **−0.0060**（过度加密反降）；BTE-only：RT 120/1 −0.0007、RB 160/3 **+0.0062**、BA 80/15 −0.0019。**同密度对比**：RT@NN-120 base(LF10) 0.9336 ≈ BTE-only 0.9330（加密后持平 → 可在 RT 补齐 CNAS 损失）；BA@NN-80 base 0.9494 < BTE-only 0.9600（仍差）→「采样密度不足是主因」假说**不获强支持**（单种子口径，谨慎）。③ **#72（RB NN-160/LF-10）因与 #73 同卡并发 CUDA OOM 丢失** → 补跑行已插 **#85**（加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`）；k×N 网格顺延 **#86/#87**。队列现状 87 行。

- **2026-09-15 傍晚（续28·全量收集 75/75 + BTE 全数据集结果（强信号）+ ⚠️ S1 基线崩溃 bug）**：① **收集**：新增快照模式（`fetch_results.py --set e2snap --allow-partial`）将 **full / CNAS-only / BTE-only × 5 数据集 × 5 种子 = 75/75 全部拉回**（`results/E-2_ablation/raw_seeds/`；sha256 入 `_sync_raw_log.csv`）。实况：#66✅(09:13)、#67✅(12:54)、密度扫描 #68–76✅、#78 CNAS-only RT/RB✅(17:44)；**#77 vanilla 运行中**（GPU0，ETA 今夜 ~21:30–22:30）、**#81 base 运行中**（GPU1，18:02 起）；#82 待跑、#83/84 网格待跑。② **BTE 全 5 数据集配对 Δ**：**(a) BTE-only − CNAS-only：auc 5/5 全正**（BA +0.0075/t=4.4、OTC +0.0097/t=8.2、RB +0.0064/t=1.7、RT +0.0109/t=7.7、WV +0.0071/t=7.4；ap 亦 5/5 正）→ **BTE 单独显著优于 CNAS 单独**；**(b) BTE-only − full**：BA +0.0036、OTC +0.0031、RB +0.0025（n.s./微正）、**WV +0.0064（t=30.8，超越 full）**、RT −0.0021（ap −0.0073/t=−3.0 显著负）→ **BTE-only 为最强单模块配置、CNAS 叠加无净益**。③ **⚠️ S1（真 DyGFormer）#79/#80 快速崩溃**：`AttributeError: 'DyGFormer' object has no attribute 'profiler'`（E-1 profiler 仪器化假定 Sign 模型接口；崩溃点=首个周期测试 epoch 5）。**修复提案**：`models/DyGFormer.py` 镜像 SignDyGFormer 加 `self.profiler = Profiler(); self.profiler.disable()`（空记录 summary/save 安全）；修复后重排 2 任务（插在 #82 后、网格前）。**待用户批准。**

- **2026-09-15 上午（续27·排期刷新 + 信号/噪声研判）**：实时（08:28）：#66 full 收尾（WV 剩 1–2 run，预计 ~09:10 完）；#67 BTE-only 剩 RB×4 + WV×5（预计 ~11:45 完）；#68–76 密度扫描随后双卡穿插（~09:10–14:30）；**#77 vanilla ~13:10 起（约今夜 23:30 完）**、#78 CNAS-only RT/RB ~13:45 起（约 18:15 完）→ **2×2 全齐 ≈ 09-16 上午**（含 #81 base 今夜 ~24:00 起、明晨 ~10:00 完）；#79/80 S1 真基线今夜—明晨；#82 明晨；#83/84 k×N 网格 09-16 白天起 → **预计 09-18/19 全部收尾**（截止 10-10 有余量）。**信号研判（回应「有的贡献像误差」）**：真信号 = BTE-only>CNAS-only（BA/OTC，t=4.4/8.2）、CNAS 叠加无益（RT ap t=−3.0）；**疑似噪声/无贡献** = RAS/RAE（±0.005 无方向）、CNAS 单种子 5/5 轻负（−0.002~−0.011，待 #78 5 种子）、BTE 在 BA/RT 的单种子微差。最终表述待 5 种子全齐定。

- **2026-09-15 上午（续26·术语澄清「基座＝w/o all」+ BTE-only vs w/o all 现状）**：用户明确：**「基座」＝ w/o all ＝ vanilla [F,F,F,F]**（续25 中曾把「基座」记为 base [F,F,T,T]，现予澄清；两套数字均有效、勿混淆）。**BTE-only vs w/o all**：5 种子待 #77（密度扫描 #68–76 后自动接续，预计今夜—明晨出数）；单种子（seed42）Δauc：BA +0.0002（平）/ OTC **+0.0085**（ap +0.0413、sF1 +0.0116）/ RT −0.0003（平）/ RB −0.0100（但 sF1 +0.0118，混合）；WV 待跑。对照：**CNAS-only − w/o all 单种子 5/5 为负**（BA −0.0061 / OTC −0.0040 / RT −0.0110 / RB −0.0018 / WV −0.0050）→ 单种子下 BTE 单独明显好于 CNAS 单独。

- **2026-09-15 上午（续25·BTE-only vs base [F,F,T,T] 现状；注：「基座」按用户口径＝w/o all，见续26）**：用户询问 BTE-only 与 base（[F,F,T,T]＝BTE+CNAS、无 RAS/RAE）对比 → **base 5 种子未跑（#81 排队中）**，暂无 5 种子配对；单种子（seed42：E-2 v3 base vs 新批 BTE-only）Δauc：BA **+0.0065** / OTC **+0.0072** / RT +0.0026 / RB −0.0058（混合、噪声内，仅参考）。**结构提示**：base 与 full 仅差 RAS/RAE（近 no-op；seed42 差距 ±0.001–0.008 无系统方向）⇒ base≈full ⇒「BTE-only vs base」的 5 种子结论预计与「BTE-only vs full」一致（BA +0.0036/OTC +0.0031 n.s.、RT −0.0021 且 ap 显著负）。

- **2026-09-15 上午（续24·BTE 中期结果已同步 Paper + 口径澄清）**：已发 `Paper`（`mb-20260915-082414-code-a734`）：BTE 中期结果（=续23 数字）+ 数据口径（**E-2 消融加种子批 #65/#66/#67**，linksign 3 分类、修复后代码、per-dataset NN/LF 最优、P1/TE、5 种子；**full 批与主表数字逐位一致**=同口径基准）+ 待补清单（vanilla/base 5 种子 → 完整 2×2 定稿）。建议 Paper 等 2×2 完成后定稿引用。

- **2026-09-15 上午（续23·BTE 中期结果：BTE-only 显著强于 CNAS-only）**：`#67` 五种子进度 BA/OTC/RT 15/15 完成（RB 1/5、WV 待跑；预计 12:00–13:00 全完）。已同步中期数据（`fetch_results.py` 新增 `--set e2now`，45 文件，sha256 入 `_sync_raw_log.csv`；新工具 `tools/verify/agg_configs.py` 按旗标聚合+同种子配对）。**5 种子（同种子配对 Δ）**：① **BTE-only − CNAS-only**：BA auc **+0.0075（t=4.4）**、ap +0.0247（t=5.7）、f1_mac +0.0235（t=2.2）；OTC auc **+0.0097（t=8.2）**、ap +0.0371（t=7.9）、sign_f1 +0.0082（t=4.7）→ **BTE 单独显著优于 CNAS 单独**；② **BTE-only − full**：BA +0.0036（t=1.7）n.s.、OTC +0.0031（t=1.4）n.s.、RT −0.0021（t=−1.8）且 ap **−0.0073（t=−3.0）显著负** → 在 BTE 上叠加 CNAS 无增益、RT 上有害（与「CNAS 裁剪丢证据」一致）；③ WV 当前仅 CNAS-only 5 种子（auc 0.9605±0.0019、ap 0.8129±0.0061）。**待补**：vanilla/base 5 种子（#77/#81）、RB/WV BTE-only、CNAS-only RT/RB（#78）→ 齐后出完整 2×2 配对。

- **2026-09-15 上午（续22·与 Paper 同步审计结论 + 批次实况）**：① 已发 `Paper`（`mb-20260915-081806-code-f689`）：早停 notice 未接线（实际判据=全 val 指标 AND）+ sign 任务 best_thr 三元写反（sign 阈值型指标 @0.5 口径；linksign 主任务无此问题）+ 写作口径提示（方法节按「全部验证指标同时不下降」表述）。② **批次实况（08:20）**：#66 full BA/OTC/RT/RB 各 5/5 完、WV 3/5（~09:30 完）；#67 BTE-only BA/OTC/RT 各 5/5 完、RB 1/5 进行中（预计 ~13:00 全完）；#65 CNAS-only 09-14 22:49 全完。③ **输出位校正**：linksign（3 分类）结果在 `saved_results/SignLinkPrediction/`、sign 在 `saved_results/LinkSign/`（fetch 工具 glob 指向正确）。④ e2s/e2c 取数分别在 #77/#78 与 #67/#81/#82 完成后执行。

- **2026-09-15 凌晨（续21·早停判据审计：`--early-stop-notice` 未接线 + sign 阈值三元写反）**：用户问询「当前实验的 notice 配置」→ 审计结论：① 队列/`run_experiments` SCRIPT_EXTRA 注入的 `--early-stop-notice`（linksign `f1_wt f1_mic ap f1_mac auc`；sign `f1_binary auc f1_weighted`；directlink `ap auc f1_binary`）**从未接线**：pydantic 字段 `early_stop_notice`（`load_configs.py:90`，描述“不提供即关注全部指标”）被解析但训练脚本从未传给 `EarlyStopping(metric_notice=…)`（该参数全仓库只出现在 `EarlyStopping.py` 内部；git 历史 cb79176 一次引入、无任何调用方）→ 实际 `metric_notice=None` ⇒ **best checkpoint 判据 = 全部验证指标同时 ≥ 各自历史最佳（AND/帕累托式）**，连续 20 轮不满足即早停（patience=20）。② **建议（待用户确认）：本轮不接线**——已跑/在跑批次全同口径，横向对比一致；比设计意图更严但一致。③ 相邻发现：`train_link_sign_prediction.py:511` `best_thr = 0.5 if hyper_param is not None else …` **三元写反** → sign 任务最终测试**恒用 0.5 阈值**（val 搜出的最优阈值被丢弃；param.json 缺失时反而会 `None.get` 崩）。linksign(3class) 无此问题（`train_sign_link_3class_prediction.py:564-566` 正确读 `hyper_parm["best_sign_thr"]/["best_exist_thr"]`）。是否修 sign 阈值 bug 待拍板（影响 sign 已有数字口径一致性）。

- **2026-09-15 凌晨（续20·图风格定调 + 体检已发 Paper）**：① 用户指示 k×N 重绘图**保持原版统一风格**——生成链：`result_collect.py`（扫 `saved_results/{SignLinkPrediction|LinkSign}/SignDyGFormer/{dataset}/*.json` → 根目录 `param-*.csv`）→ `analysis/param-graph.py`（seaborn `crest`、`fmt=".4f"`、每数据集每指标一张、输出 `figs/{task}-{dataset}-img.pdf`）；现有 10 张旧图（2026-06-24 渲染）在 `figs/`（如 `linksign-WikiRfA-img.pdf`、`Sign-BitcoinAlpha-img.pdf`），重跑后原地替换。② **BTE 离线体检结果已发 `Paper`**（`mb-20260914-220430-code-9815`；含「乐观上界」口径警示与引用注意事项）。

- **2026-09-15 凌晨（续19·k×N 超参网格重跑入队＝方案甲）**：用户追问「NN 是否也要超参实验」→ 确认这正是已批的**方案甲**（Paper 下单 `fd4c` ②）：论文 k×N 热图底层网格 = `run_experiments.py -t parameter`（`PARMA_GROUPS`：LF {1,3,5,10,15,20} × NN {10,15,20,40,60,80,100} = **42 点/图**）；现有 `param-linksign.csv` / `param-sign.csv`（2026-05）为**修复前口径**（如 BA linksign 0.944–0.962），与修复后主表不同源 → 必须重跑。**已入队两行（队尾，入队文件 `tools/queue/grid_kN_20260915.txt`，tasks.txt→84 行）**：`-s linksign -t parameter -m SignDyGFormer`（210 runs）与 `-s sign -t parameter -m SignDyGFormer`（210 runs）；**单种子 42、P1/TE、加速默认**——与主表同代码口径。排期：前面批次（ⓑ→密度扫描→vanilla→CNAS-only→S1→base→RT 探边）跑完后自动接续（约 09-16），**预计 3.5–4.5 天双卡**完成。下游：重生成 `param-*.csv` + 重绘 k×N 图（脚本化、对齐库内风格）。**密度发现加持**：NN≡序列长度上限 ⇒ 该网格同时构成「采样密度敏感度」证据（BTE/CNAS 叙事的核心材料）。

- **2026-09-14 深夜（续18·BTE 离线信号体检）**：新增 `tools/verify/bte_signal_check.py`（纯 CPU、读列裁剪、每数据集抽样 1200 条测试期真实边；忠实复刻 BTE 计数：pos/neg 出现对计数 + 去重投票；对照多数类/直接历史预测器；**乐观上界口径**：全历史截近 K=100 求交，模型侧还要经 CNAS 窗口 + NN 截断）。**结果**：加权证据 AUC — **OTC 0.831 / BA 0.724 / RB 0.707 / WV 0.683 / RT 0.549**（平衡准确率 0.556–0.750；覆盖率 42%–80%）；BA 分桶单调递增（[2-4] 0.860 → [5-19] 0.911 → [20-99] 0.923）。**结论：三元平衡证据在数据中真实存在（4/5 显著），BTE 的理论正确性成立；模型侧没吃到 → 归因于覆盖（模型可见面 ≪ 体检面）+ 聚合尺度/重复膨胀 + 冗余/训练动力学，而非理论错误。** 旁证：直接历史预测器在覆盖子群上很强（RB 0.908@cov 0.463、RT 0.777@cov 0.321）→ RAE 的 direct 定位有据。

- **2026-09-14 深夜（续17·采样密度扫描入队 + BTE 门控预留接口）**：① 用户批准「密度验证优先推进」。查证两个**非结构**旋钮：**NN `--num-neighbors`＝序列长度上限**（`load_configs:212` `max_input_sequence_length≡num_neighbors`；超限保留最近 NN−1＋自身）与 **LF `--common-neighbors-look-forward`＝CNAS 窗口半径 k**（CNAS-on 独有）。**扫描（单种子 42、linksign、`@` 原生命令）插至队首（#68 起，BTE-only #67 之后）**，清单版本化于 `tools/queue/sweep_density_20260914.txt`：RT {base@LF10・base@NN120/LF10・BTE-only@NN120}、RB {base@LF10・base@NN160/LF10・BTE-only@NN160}、BA {base@LF20・base@NN80/LF20・BTE-only@NN80}＝9 runs（≈2–4h，双卡分流）。判读：**base@高密度≥vanilla → 密度不足（修法=只调 NN/LF）；BTE-only≥vanilla 而 base 上不去 → CNAS 设计问题；都上不去 → BTE 弱**。② **BTE 门控预留接口（用户指示：预留启用、默认禁用，后续或有向图用）**：`SignDyGFormer` 第 5 通道加 sigmoid 门（`balance_theory_gate_raw` 初始 −6≈0.0025；仅「编码器开＋门控开」时建模，否则 None 零副作用）；CLI `--module-balance-theory-gate`（默认 False）；结果名加 `.GATE` 防覆盖；两训练脚本透传（提交 `e344c76`；冒烟：仅 +1 参数、初值 0.00247；服务器已 pull）。

- **2026-09-14 深夜（续16·返修期不动结构；BTE-only 拆解优先入队）**：用户指示：① 返修期**避免结构改动**（门控=改模型定义→需全量重跑，实施与否待评估；已说明门控原理与代价）；② **优先推进 BTE-only（[F,F,T,F]）实验**。落地：`run_experiments.py` MODULE_GROUP 新增第 7 行 `[F,F,T,F]`（BTE 开、CNAS 关；**仅实验组合表新增，零架构风险**）；入队 `--module-idx 6 -e`（5 数据集×5 种子＝25 runs）并**插至队首位置 #67**（原 #67–#72 顺延为 #68–#73）——GPU 空出即派。设计：与已排队列构成 **CNAS×BTE 2×2 全因子**（vanilla=[F,F,F,F]・CNAS-only=[F,F,F,T]・BTE-only=[F,F,T,F]・base=[F,F,T,T]，同种子配对）→ 可直接分离「CNAS 采样裁剪」与「BTE 证据通道」两个嫌疑。

- **2026-09-14 晚（续15·用户放行：S1 真基线批 + 消融 base 加种子入队）**：用户指示「开始」→ 按建议方案入队（队列 **#69–#72**，`tasks.txt`=72 行）：**#69 `-s sign -m DyGFormer -e -r BA OTC WV`（RT/RB ×5 种子＝10 runs）**、**#70 linksign 同款（10 runs）**＝**S1 最小档（真基线 20 runs）**，用于替换作废的旧「DyGFormer」行（溯源见续14）；**#71 base 行加种子（[F,F,T,T]×5 数据集×5 种子＝25 runs）**；**#72 +RAS/+RAE RT-only（2 配置×5 种子＝10 runs）**。合计 **55 runs**；排期：#67/#68（ⓑ）后自动接续（预计 09-15 白天出数）。注：DyGFormer 为仓库内置模型（`models/DyGFormer.py`），协议与主表一致（NN/LF 选参、P1·TE、加速开）；出炉后主表基线行/ Avg.Rank / R2-8a 以新数字重算。

- **2026-09-14 晚（续14·DyGFormer 行溯源发现 + 波二 pure 后端解锁）**：① **Paper 发现并问询**（`mb-20260914-181200-paper-5518`）：主表「DyGFormer」行与旧消融「w/o all」行逐位相同。**本地溯源（回执 e532）**：该行实为**旧消融（SignDyGFormer 模块组合）行**——sign 与根目录 `A-result.csv` 的 `D,D,D,D`（全关）行精确吻合（OTC `0.8484/0.8081`、RT `0.7046/0.9190` 逐位；BA F1=0.7314 同为 DDDD、其 AUC 0.8042 实为 `aaa.result.json` BA-DyGFormer run[0]；RB `0.6444/0.8627` 实为 `D,D,D,E` 行；WV 疑旧版本/手改）；linksign 与 `A-linksign.csv` DDDD 行吻合（OTC/RB/RT 逐位、BA F1_W 逐位；WV 不吻合）。**结论：主表基线标名有误，该行不可用，需重跑真基线**（`run_experiments.py -m DyGFormer`，同协议 5 种子；旧数字叠加修复前时代不可比）。② **合并方案已回 Paper**：S1 最小档 20 runs ≈4–6h / 满档 50 runs ≈10–15h（可选加档 +25 runs）；消融补种子建议优先 **base（25 runs）**，+RAS/+RAE 折中 RT-only（10 runs）；等 Paper 圈档位 + 用户放行后入队。③ **Baseline pure 后端解锁**（`mb-20260914-184502-baseline-9a9e`）：DyG-Mamba 端到端已跑通（pure 后端，无需 mamba-ssm/causal-conv1d 编译；建议 `--batch_size 64`）；已回执采纳（d820），343MB 官方 CUDA 核 wheel 暂不需；请其确认 pure 最小依赖 → dygmamba env 可按「无 mamba 编译」收官，**GitHub 轮子阻塞彻底解除**。④ 队列：ⓑ #65/#66 双卡运行、#67/#68 排队（预计明早齐）。

- **2026-09-14 晚（续13·OTC sign 换装落定 + 口径统一）**：① 用户拍板 **OTC sign 行换装 NN-40/LF-15**：**0.8775±0.0054**（旧 NN-60/LF-10 = 0.8696±0.0048；配对 Δ+0.0079、t=3.16、5/5 全正）——sign 主表唯一变更点（RT/RB/BA/WV 保持）。② **std 口径统一为 pstd（ddof=0，与主表/snapshot_results 一致）**：`nh5_report.py` 已改（描述统计 pstd；配对 t 仍用样本 sd）；两个 ddof=1 表述已修正。③ 工具/归档：`snapshot_results.py` OTC sign glob→NN-40.LF-15、`fetch_results.py` MAIN_SIGN_OTC 同步换格；新配置 5 个 raw JSON 已归入 `results/main_tables/raw/sign/BitcoinOTC/`（与旧 NN-60/LF-10 并存留档）。④ **w/o all（vanilla）现状**：单种子已有（E-2 ⓑ 表：BA 0.9617 / OTC 0.9671 / WV 0.9670 / RT 0.9340 / RB 0.9334，auc）；**5 种子正跑（#66 full / #67 vanilla / #65+#68 CNAS-only 全 5 数据集）**，预计 09-15 早出数。

- **2026-09-14 晚（续12·队列事故修正 + #61/#62 复核出数 + 波二汇报 Baseline）**：① **队列事故**：误把 `-r/--ignore-dataset`（**排除**语义）当「选集」用→ ⓑ 三行全写错：#63/#64 全 skip、#65 跑成 BA/OTC/WV；已重排 **#66 full(全5数据集)/#67 vanilla(全5)/#68 CNAS-only RT/RB**（#66 已于 18:08 GPU1 开跑）；#65 保留为 **CNAS-only 全 5 数据集**补充（多占 ~2-3h GPU）。② **#61/#62 五种重复核完成并同步**（nh5 集扩至 32/32，`fetch_results.py` 已更新；sha256 入 `_sync_raw_log.csv`）：**WV 20/20 = 0.7968±0.0022 vs 现行 0.7956±0.0020（pstd），配对 Δ+0.0012（t=0.69）n.s. → 不更新**（seed42 的 +0.0065 被证伪）；**OTC 40/15 = 0.8775±0.0054 vs 现行 0.8696±0.0048，配对 Δ+0.0079、5/5 全正、t=3.16（df=4，≈p<0.05）→ 边缘显著**。`nh5_report.py` 扩展了配对检验段。③ **R1-4 波二现状+卡点已汇报 `Baseline`**（`mb-20260914-180951-code-e616`）：scadyg 就绪、dygmamba 缺 mamba-whl（拟经 git 通道传官方 whl）→ 求其评方案/替代/是否先跑 ScaDyG。

- **2026-09-14 午后（续11·R2-11 P 热力图交付）**：按用户指示「参考现有热力图配色与布局 + 脚本化」完成 **P 敏感性热力图**：新增 `tools/fig/gen_p_patch_heatmap.py`（读 `results/E-3_patch/raw/`，5 数据集 × P{1,3,5,7}，AUC 主面板 + $F1_{wt}$ 副面板；**seaborn `cmap="crest"`、`fmt=".4f"`、显示名 WikiRfA/RedditTitle/RedditBody、14pt 轴标签——与 `analysis/param-graph.py` 既有热力图同款**；运行时与 `E3_patch_summary.md` 84/84 交叉校验通过）→ 产物 `figures/fig_p_patch_heatmap.{png,pdf,csv}`。**结论：无一致趋势**（各数据集 P 跨度 0.0025–0.0135，均 ≤ 种子噪声；默认 P=1 有据；仅 OTC 在 P=3/5 高 +0.0060，单种子口径）。**⚠️ 提出：`results/E-3_patch/E3_patch_summary.csv` 为旧伪交集数据（如 RB P1=0.9610 vs 修正后 0.9239），勿引用**（脚本已绕开）。另修 `.gitignore`：`fig/` 规则误伤 `tools/fig/`，已加白名单例外；**图产物（png/pdf/csv）一律不入库**（用户 09-14 指示：二进制不便管理；脚本可确定性复现），先前入库的 PNG 已自版本库移除。

- **2026-09-14 午间（续10·third_party 迁移 + Paper 回复 + env v2 推进）**：① **third_party 源码入服务器**：GitHub 从服务器不可达 + 浅克隆直推被拒（`shallow update not allowed`，`receive.shallowUpdates` 未生效）→ 改用 **vendor 快照仓**（本地 `git archive` + 单 commit 标注上游 URL/commit → 推新裸库 `~/git/DyG-Mamba.git`/`ScaDyG.git` → 服务器 clone 入 `ext_baselines/third_party/`；快照 `897721a`/`ca9f555`）。② **env v2**：dygmamba（numpy1.26.4 + setuptools69.5.1 pin 后）正编译 causal-conv1d/mamba-ssm；scadyg 重建至收尾（PyG 精确轮子 + dgl 已装）。③ **`Paper` 行动项已回**（`mb-20260914-124215-code-bd25`）：ⓑ编排=3 任务（full/vanilla 各 25 runs + CNAS-only 10）≈ 双卡 8–13h；基线补种子 **go**（S1 in-repo：最小 20 / 满档 50 runs；S2 SEMBA/FreeDyG/TiDFormer/SiGAT 建议波三立项）；p 值口径 4 条（单样本 t + 效应量 + bootstrap CI + 表格标注）。④ #61/#62 进行中（WV 3/5、OTC 2/5）。\n\n- **2026-09-14 午间（续9·波二 env 修复 v2 + third_party 通道）**：v1 构建实测暴露两问题：① **dygmamba**：py3.10 env 里 numpy 被升到 **2.2.6** + setuptools 83（无 pkg_resources）→ torch2.1 构建隔离失败（exit 14）⇒ **v2：pin `numpy==1.26.4`、`setuptools==69.5.1`**，nvcc（conda cuda-nvcc=11.8 已在 env）下源码构建 `causal-conv1d==1.4.0 / mamba-ssm==2.2.2`、`TORCH_CUDA_ARCH_LIST=7.5`、`--no-build-isolation`。② **scadyg**：PyG 轮子未匹配→源码编译（无 nvcc 将失败）⇒ 实测 data.pyg.org 可用，**v2：重建 env + 精确 pin 轮子**（scatter2.1.0+pt112cu116 / sparse0.6.16 / cluster1.6.0 / spline-conv1.2.1）+ numpy1.23.4 先置。③ **third_party 源码不在服务器**（gitignore 未传；GitHub 从服务器不可达）⇒ 拟走 **git 通道**：本地浅克隆仓（DyG-Mamba@ce54319 / ScaDyG@28ca94a）URL 直推新裸库 → 服务器 clone 入 `ext_baselines/third_party/`。另：`Paper` 行动项（ⓑ加种子/基线5种子评估/p值口径）已读，回复与编排进行中。

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
