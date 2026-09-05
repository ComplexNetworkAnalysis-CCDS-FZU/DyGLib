# NEUCOM-D-26-13975 完整审稿意见归档

> 归档日期：2026-09-05
> 稿件：SignDyG: Balance-Theoretic Evidence Encoding for Continuous-Time Dynamic Signed Graph Prediction（Neurocomputing）
> 原始评审日期：2026-07-21；修改截止：2026-10-10
> 用途：与导师逐条讨论应对方案（Agent A 负责正文/理论/回复信；Agent B 负责代码/实验）
> 配套：`EXPERIMENT_PLAN.md`（实验计划）、`PROGRESS.md`（进度）、`REPORT_TO_ADVISOR.md`（导师汇报）

---

## 审稿人 #1（共 6 条意见 + 文献补充要求）

### R1-1【正负证据通道的极性保持 / 通道交换不变性】
> 原文：In Eq. (13), the positive- and negative-evidence channels appear to be processed by the same function and then summed. This operation may be invariant to swapping the two channels. How does BTE preserve sign polarity and distinguish opposite evidence configurations?

- **关注点**：正/负证据通道用同一函数处理后相加，可能对两通道互换不变 → BTE 如何保持符号极性、区分相反证据配置？
- **应对方向**：理论分析（Agent A）；必要时合成实验佐证
- **状态**：待讨论

### R1-2【式(9)笛卡尔积的时间有效性 / 时间衰减对比】
> 原文：Eq. (9) takes the Cartesian product of all historical (u,w) and (v,w) interactions. Why should every pair of historical interactions constitute valid triadic evidence, particularly when the two events may be widely separated in time? Comparisons with temporally matched, most-recent, or time-decayed evidence construction are needed.

- **关注点**：任意两条历史交互配对是否都构成有效三元组证据（尤其时间间隔大时）？需要与时间匹配 / 最近 / 时间衰减证据构造对比
- **应对方向**：**E-4 时间衰减对比**（已实现：指数衰减 $e^{-\lambda\Delta t}$，A=陈旧度/B=事件间隔两种 Δt 可切换，`--time-decay-lambda/--time-decay-gap-mode`）
- **状态**：代码 ✅；实验 ⏳ 运行中

### R1-3【"分离正负记忆通道导致 3/4 平衡配置不可恢复"缺乏理论支撑】
> 原文：The claim that separated positive/negative memory channels make three of the four balance configurations "non-recoverable" is not theoretically established. Separate channels followed by nonlinear fusion may retain cross-sign information... The authors should provide either a formal non-identifiability analysis or controlled synthetic experiments to support this claim.

- **关注点**：该论断需正式不可辨识性分析或受控合成实验支撑
- **应对方向**：形式化分析（Agent A）或合成实验（如需 Agent B 可补充）
- **状态**：待讨论

### R1-4【相关工作缺失近期文献】
> 原文：related-work omits substantial recent literature... including DyG-Mamba, UniDyG, recent multimodal dynamic link-prediction methods, and ScaDyG [1-4]. Although some do not specifically address signed graphs, they are relevant to temporal modeling, scalability, and representation-learning claims.

- **参考文献**：[1] DyG-Mamba: Continuous State Space Modeling on Dynamic Graphs；[2] UniDyG: A Unified and Effective Representation Learning Approach for Large Dynamic Graphs；[3] Unlocking Multi-Modal Potentials for Link Prediction on Dynamic Text-Attributed Graphs；[4] ScaDyG: A New Paradigm for Large-Scale Dynamic Graph Learning
- **应对方向**：Agent A 补写相关工作 + 讨论
- **状态**：待讨论

### R1-5【对 CNAS/RAS/RAE/CNE/BTE 分别消融 + 交互效应】
> 原文：Please conduct separate ablations of CNAS, RAS, RAE, CNE, and BTE, and further analyze their pairwise and higher-order interaction effects. The current coupled ablations do not clearly isolate the contribution of each component.

- **应对方向**：**E-2 增量式消融**（已就绪：基线→+CNAS→+RAS→+RAE→+BTE；CNE 属 DyGFormer 骨架不单独消融；link&sign 任务；RedditTitle + WikiVote@20000）
- **状态**：代码 ✅；实验 ⏳ 运行中
- **注**：⚠️ RAE 曾有 `np.append` 未赋值 bug（已修复），旧消融数据中 RASE 列无效，勿引用

### R1-6【效率与可扩展性：时间/显存/参数量/不同规模】
> 原文：Please report training and inference time, peak GPU memory consumption, parameter counts, and scalability results on graphs of varying sizes. Comparisons should be conducted under the same hardware and software settings.

- **应对方向**：**E-1 效率数据**（已实现：训练/推理时间、参数量、峰值显存自动写入结果 JSON；RedditTitle@20000）
- **状态**：代码 ✅；实验 ⏳ 运行中

---

## 审稿人 #2（共 9 条意见）

### R2-1【加性融合 vs 拼接的理论推导】
> 原文：The paper claims summing positive/negative evidence vectors outperforms concatenation with fewer parameters, but only provides minor empirical differences without mathematical explanation. Derive the feature space difference between additive fusion and concatenation: analyze why additive aggregation naturally matches balance theory's additive evidence accumulation logic, and prove concatenation introduces redundant dimension space without semantic gain.

- **应对方向**：数学推导（Agent A）
- **状态**：待讨论

### R2-2【采样算法复杂度的理论上界】
> 原文：The paper empirically verifies neighbor activity is 2.5-7× higher than self-activity but lacks theoretical complexity bounds of the sampling algorithm. Derive the upper bound of sampling sequence length and time complexity of common neighbor retrieval; prove the strategy avoids exhaustive second-order traversal and ensures linear complexity relative to event volume.

- **应对方向**：理论推导（Agent A，可与 E-1 实测数据互相印证）
- **状态**：待讨论

### R2-3【符号类别不平衡下全局平衡率的偏置 / BTE 局部计数的缓解】
> 原文：The paper only observes this phenomenon but does not analyze how extreme sign skew distorts balance signal extraction. Supplement a theoretical subsection discussing the bias of global balance rate under class imbalance, and explain how BTE's per-triad local counting mitigates this skew compared to global aggregation baselines like SEMBA.

- **应对方向**：理论小节（Agent A）；如需数据佐证 Agent B 可跑分析
- **状态**：待讨论

### R2-4【Patch-wise 自注意力的收敛稳定性与时间依赖保持】
> 原文：Analyze the convergence stability of patch-wise self-attention; explain why non-overlapping patches retain temporal dependency without information loss.

- **应对方向**：理论分析（Agent A）
- **状态**：待讨论

### R2-5【DynamiSE / DySDGNN 缺失的公平性处理】
> 原文：The paper only cites SEMBA as continuous-time signed baseline, while DynamiSE, DySDGNN are excluded due to no open code. To strengthen fairness, implement simplified continuous-time event-stream versions of DynamiSE/DySDGNN for comparison, or add detailed analytical discussion of their structural disadvantages.

- **应对方向**：倾向"详细分析讨论其结构劣势"（不实现；Agent A 负责）；是否实现待与导师讨论
- **状态**：待讨论

### R2-6【噪声鲁棒性实验】
> 原文：Real-world signed networks contain three common noise types: random sign flip, spurious positive/negative edges, missing historical interactions. The current experiments only test clean datasets; design three gradient noise levels for all five datasets, compare SignDyG against SEMBA/DyGFormer under noise to validate BTE's anti-noise capacity for triadic balance signals.

- **关注点**：3 种噪声类型 × 3 梯度级别 × 5 数据集；对比 SEMBA/DyGFormer
- **应对方向**：**E-7 可插拔噪声模块**（已实现：`utils/noise.py`，sign flip，可迁移 SEMBA 仓库，双方同 seed 保证公平）。**本轮计划不执行**（时间考虑），是否补充跑待与导师讨论
- **状态**：代码 ✅（模块保留）；实验 ⛔ 本轮不做（待讨论）

### R2-7【异配图基准测试】
> 原文：All five datasets are homophily-biased signed networks. Add standard heterophilous signed benchmarks (e.g., Signed-Cornell, Signed-Texas) to test SignDyG generalization, and analyze the performance gap of BTE under weak triadic balance signals.

- **应对方向**：**E-6**——本轮不做（数据需另下载、加管线）；是否补做待与导师讨论
- **状态**：⛔ 本轮不做（待讨论）

### R2-8【p 值显著性（10 种子）+ Patch size P 消融】
> 原文：Compute p-values for Reddit Sign Prediction and all Link&Sign Prediction metrics with 10 repeated experimental seeds; mark statistically significant performance gaps in all result tables. Hyperparameter analysis only covers k (lookback) and N (history length), ignoring patch size P which directly controls Transformer input length. Add heatmap ablation of P across all datasets to quantify its impact on prediction AUC/F1.

- **关注点 1**：10 个重复种子算 p 值并标注显著性 —— ⚠️ **当前计划为 5 种子**（与审稿人 10 种子不一致，待与导师确认是否按 10 种子执行）
- **关注点 2**：Patch size P 消融热力图 —— **E-3 已就绪**（P∈{1,3,5,7}，WikiVote + RedditBody@20000，link&sign）
- **状态**：E-3 代码 ✅ 实验 ⏳；E-5 种子数 ⚠️ 待讨论

### R2-9【空间复杂度 / 模块级峰值显存对比】
> 原文：The theoretical time complexity formula is provided but lacks space complexity analysis; calculate peak memory consumption of sampling, BTE, Transformer modules for large-scale Reddit datasets, compare memory overhead with SEMBA/TGN.

- **应对方向**：E-1 已含整体峰值显存 + profiler 各模块时间；模块级显存拆解与空间复杂度理论分析待补充（Agent A + 必要时 Agent B 加测）
- **状态**：部分 ✅，待讨论

---

## 应对映射总表

| 意见 | 内容 | 承担 | 本轮执行 |
|---|---|---|---|
| R1-1 | 极性保持/通道互换不变性 | Agent A（理论） | 讨论 |
| R1-2 | 时间衰减证据构造对比 | Agent B（E-4） | ✅ 运行中 |
| R1-3 | 不可恢复性理论/合成实验 | Agent A | 讨论 |
| R1-4 | 补相关文献 | Agent A | 讨论 |
| R1-5 | 分模块消融 + 交互 | Agent B（E-2） | ✅ 运行中 |
| R1-6 | 效率（时间/显存/参数/规模） | Agent B（E-1） | ✅ 运行中 |
| R2-1 | 加性 vs 拼接理论 | Agent A | 讨论 |
| R2-2 | 采样复杂度上界 | Agent A | 讨论 |
| R2-3 | 类别不平衡偏置理论 | Agent A | 讨论 |
| R2-4 | Patch 注意力收敛/时间保持 | Agent A | 讨论 |
| R2-5 | DynamiSE/DySDGNN 处理 | Agent A（分析讨论） | 讨论 |
| R2-6 | 噪声鲁棒性 | Agent B（E-7 模块已备） | ⛔ 暂不跑，讨论 |
| R2-7 | 异配图（Signed-Cornell/Texas） | Agent B（E-6） | ⛔ 暂不做，讨论 |
| R2-8a | p 值（**10 种子**） | Agent B（E-5） | ⚠️ 计划 5 种子，确认 |
| R2-8b | Patch size P 消融 | Agent B（E-3） | ✅ 运行中 |
| R2-9 | 空间复杂度/模块显存 | Agent A + B | 部分 ✅，讨论 |

## 与导师讨论的开放项（待定）

1. **R2-8a 种子数**：审稿人要求 10 种子，当前计划 5 种子 → 是否按 10 种子执行（影响 E-5 耗时，约 ×2）？
2. **R2-6 噪声实验**：是否本轮补跑（模块已就绪；需定数据集/级别/范围 train-only 或全数据）？
3. **R2-7 异配图**：是否补做（需下载 Signed-Cornell/Signed-Texas 并加数据管线）？
4. **R2-5 DynamiSE/DySDGNN**：仅文字分析，还是实现简化版（工作量大）？
5. **R1-3 / R2-1 / R2-2 / R2-3 / R2-4**：理论类条目是否需要 Agent B 提供数据/合成实验佐证？
