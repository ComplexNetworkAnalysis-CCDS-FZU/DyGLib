# SignDyG 修订实验进度记录（NEUCOM-D-26-13975）

> **维护者**：`Code`（代码与实验；历史代号 B）
> **用途**：所有 Agent（含 `Paper` 论文撰写）通过本文件了解最新进度。
> **规则**：每次实验状态变化立即更新本文件，并通过 git 同步（本地 push → 服务器 pull）。
> 提交截止：2026-10-10。

## 全局状态（2026-09-11 更新）

| 项目 | 状态 | 说明 |
|---|---|---|
| **Agent 代号启用** | ✅ 2026-09-11 | 协作体系改用角色代号：`Paper`/`Code`/`Baseline`/`Perf`（Rust 加速，新加入）；注册表 `docs/AGENTS_REGISTRY.md`；历史条目保留字母（A=Paper、B=Code、C=Baseline） |
| **Perf 工作区** | ✅ 2026-09-11 | `D:\codes\SignDyG-Perf`：DyGLib 热点内核**参考实现**（K1 采样链 / K2 BTE，numpy，含全部修复语义）+ 8 场景 golden fixtures + 上游逐位对照全绿 + 基线基准/回归脚手架；`TASKS.md`（M0–M5，双闸门：bit-exact + ≥10×）；⛔ **服务器禁区**（未经用户逐次许可禁止任何服务器接触，含自动 agent） |
| **Perf 进展** | ✅ 2026-09-11 | M0–M2 完成：Rust 工具链（GNU）+ K1/K2 内核（bit-exact 全绿；K1 25–27×、K2 15.5×；clippy/fmt 门禁绿）；M3 进行中（batch 入口/GIL）。详见 HANDOFF「给 Code」Perf 条目；⚙️ 待同步：CN 伪交集修复后 K1/fixtures 需更新（见 HANDOFF 行动项） |
| **Agent C 加入** | ✅ 2026-09-09 | `Baseline`（原 C）= DynamiSE/DySDGNN 复现（R2-5 ①），工作区 `D:\codes\DynamiSE_DySDGNN_repro`，权威 = `IMPLEMENTATION_SPEC.md`；结果登记 HANDOFF/PROGRESS |
| 服务器 | ✅ 可用 | 2026-09-01 恢复访问；**访问纪律（09-11）：唯一通道 = `Code`（须用户逐次许可），其他 agent（含自动）禁止接触** |
| 代码同步 | ✅ 完成 | 已推送 `sign-adoption` 分支至服务器裸仓库并 clone；`6f72c2c` 已同步 |
| 数据就绪 | ✅ 完成 | `server_setup.sh` 已执行，WikiVote tail20000 已生成 |
| 环境安装 | ✅ 完成 | conda env `gc`（torch 2.2.2） |
| **GPU 驱动** | ✅ **已修复（2026-09-06）** | DKMS `nvidia/595.84` 已装（内核 6.8.0-124/138）；`torch.cuda.is_available()=True`，设备 `NVIDIA GeForce RTX 2080 SUPER`；fedsa 免 sudo 可访问。详见 `docs/GPU_DRIVER_INSTALL.md`（实际装 595.84，非计划 590） |
| 耗时预期 | ✅ 可切 GPU 口径 | 自 2026-09-06 起新实验可按 GPU 估算；E-3 及此前日志仍为 CPU 耗时 |
| **实验队列** | ✅ 运行中（2026-09-11 起） | `tools/queue/queue_daemon.sh` 常驻守护：探测空闲 GPU（lock PID + 显存）并自动派发 `tasks.txt` 中的任务；日志 `tools/queue/queue.log`；自检 `bash tools/queue/selftest.sh` 全部通过；**追加任务 = 往 `tasks.txt` 加一行**（不含 `-g`） |
| E-7 噪声 | ⛔ 本轮不做 | 模块 `utils/noise.py` 已实现保留 |

## 实验状态总览

| 实验 | 任务 | 数据集 | 状态 | 结果路径 / 备注 |
|---|---|---|---|---|
| E-1 效率 | sign | RedditTitle@20000 | ✅ 可提取 | E-5 sign seed42 已含 4 项效率数据，待汇总 |
| E-2 消融 | linksign | **全部 5 数据集** | 🔄 **重跑中（伪交集修复版；旧 20/20 作废）** | 旧（含伪交集）数据仅存档；重跑后更新下方全量表与汇总 |
| E-3 Patch | linksign | WikiVote@20000 + RedditBody@20000 | ✅ **完成（CN 修复版，2026-09-12）** | 8/8：RB AUC 0.9149–0.9284（无单调趋势）、WV 0.9594–0.9619（近持平）→ patch 不敏感、默认 P=1 有据；已同步并替换 `results/E-3_patch/`（汇总已更新） |
| E-4 时序 | linksign | WikiVote@20000 | ✅ **完成（CN 修复版，2026-09-12）** | TD 0.9565 vs TE 0.9607（ΔAUC −0.0042、ΔAP −0.0120、ΔsignF1 −0.0118）→ 同配下 TE 略优、默认 TE 保留；汇总 `results/E-4_time_decay/E4_time_decay_summary.md` |
| E-5 显著性 | sign + linksign | RedditTitle@20000 | ✅ **完成（CN 修复版，2026-09-12）** | linksign auc **0.9391±0.0007**（ap 0.7184±0.0004、sF1 0.8779±0.0077）；sign auc **0.6746±0.0028**（ap 0.9401±0.0019）；汇总已替换；p 值待基线同口径数据（基线轨见 HANDOFF 09-12） |
| 主表重跑 | sign + linksign | 5 数据集 | 🔄 **进行中（CN 修复版队列）** | sign RT/RB 已同步（auc RT 0.6746±0.0028 / RB 0.6117±0.0354）；linksign RT/RB 运行中；WV/BA/OTC 排队；基线模型不受 CN 修复影响（Baseline 线独立） |
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

## E-2 修复后消融 全量结果（2026-09-11 17:01 完成，GPU，linksign，AUC）

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

- **2026-09-12（午）**：**E-5 10/10 完成并同步**（linksign auc 0.9391±0.0007、sign auc 0.6746±0.0028；`E5_summary.md` 已更新；同时构成主表 RedditTitle 行）。**基线轨启动**：拟把 DynamiSE/DySDGNN（Baseline 复现）及后续方法编排进连续队列——已向 `Baseline` 发**接入汇报需求**（HANDOFF 09-12）；已核实前置：服务器旧拷贝为 09-09 tar+scp（无 git，需 git 化重建）、`gc` 环境缺 `torchdiffeq`（待用户许可后安装）；暂存任务行 `tools/queue/tasks.baseline.staged.txt`。
- **2026-09-12（上午）**：**E-3/E-4 CN 修复版重跑完成并同步归档**（E-3 8/8、E-4 1/1，sha256 见 `results/_sync_raw_log.csv`；汇总已替换）；**主表 sign RT/RB 5 种子完成同步**（auc RT 0.6746±0.0028 / RB 0.6117±0.0354）；linksign RT/RB 运行中。⚠️ **服务器 NVML 故障**：无人值守升级（06:35，新内核 6.8.0-138 + 驱动用户态 595.91）导致 `nvidia-smi` 报 driver/library mismatch（内核模块 595.84 未重载）；**CUDA 训练不受影响**（新进程 `is_available()=True`、任务正常）；建议本轮队列完成后安排重启修复（待用户定）。
- **2026-09-11（晚）**：用户决策**修复 CN 伪交集**（assume_unique→真交集）；受影响实验全量重排（队列 15 条，含 @ 原生命令支持）；旧结果作废，报告以重跑后为准。**22:20 修复版已部署至服务器（pull→`9d658a9`），本地/服务器双层语义校验通过（keys=[2]）；新队列守护进程 pid=239314 启动，task#1 E-4 TD（GPU0 pid=239335）与 task#2 E-3 RB（GPU1 pid=239435）已上卡。**
- **2026-09-11**：E-2 20/20 + E-5 10/10 完成（**伪交集旧语义，已作废**，重跑排程序列中）；归档同步本地（sha256 记录）；汇总表落盘（数字已被重跑取代）；Perf M0–M2 完成（加速内核双闸门初步达成）。
- **2026-09-01**：服务器恢复；代码推送至服务器；`server_setup.sh`（数据就绪脚本）+ 本进度文件建立；E-7 确定本轮不执行。
- **2026-08-20**：E-1~E-7 全部代码实现完成（时间衰减、效率测量、消融/patch 脚本、统计脚本、噪声模块、RAE bug 修复），本地 CPU 冒烟测试全部通过；`EXPERIMENT_PLAN.md` 运行计划定稿。
