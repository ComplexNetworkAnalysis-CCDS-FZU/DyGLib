# 论文权威定义摘录：RAS 与 RAE（供代码对齐，2026-09-09）

> 摘录人：Agent A（论文侧）｜来源：`D:\Sign_DygFormer\main.tex`（NEUCOM-D-26-13975 修订稿，2026-09-09 现行版）
> 用途：Agent B 的代码对齐基准。背景（用户 09-09 定案）：**以论文定义为准，跳过 git 溯源**；B 不再自行解读论文 tex。
> 范围：RAS = 采样侧 repeat-aware（§3.2）；RAE = 编码侧 BTE 内 "direct evidence" 分量（§3.3.2）。

## 0. 命名与术语（论文用法）
- **CNAS&RAS** = 论文对**采样策略**的合称（§1 引言、§3.1、§3.4）。§3.4 有高亮句（方案修订后）：*"our contributions reside in the CNAS\&RAS sampling strategy (Section~3.2) and the BTE encoding module (Section~3.3)"*。
- **RAE 不是独立顶层模块**：是 **BTE（§3.3.2）** 内部两个证据源之一 —— "Direct evidence: repeat-aware encoding (RAE)"，与 "Indirect evidence: balance-theoretic effects"（共同邻居三连乘积）并列。
- 实验掩码位序 (RAS, RAE, BTE, CNAS) 中：RAS = 采样侧（锚点 R\_{*}）；RAE = 编码侧（BTE 的 direct 分量）。

## 1. RAS（采样侧）权威定义 — §3.2 Neighbor Sampling

### 1.1 定位与动机（原文）
- §3.2 两机制 itemize："**Repeat-aware sampling** captures direct interaction recurrence between $(u,v)$, preserving long-term relational stability patterns."
- §1 引言（CNAS&RAS 作为 supporting mechanisms）："*repeat-aware anchoring locates the exact historical positions where the target node pair has interacted before*"。

### 1.2 Repeat-Aware Node（§3.2 正式定义）
$$
\mathcal{R}_u(t) = \{\, (v, t') \mid (u, v, t') \in \mathcal{E},\ t' < t \,\},\qquad
\mathcal{R}_v(t) = \{\, (u, t') \mid (u, v, t') \in \mathcal{E},\ t' < t \,\}
$$
u 侧重复锚点 = u 的历史中与 v **直接交互**的位置（含时刻）；v 侧对称。
⇒ **触发条件即 $v \in \mathcal{N}_u(t)$（或 $u \in \mathcal{N}_v(t)$），非自环** —— 与 `DESIGN_RAS_RAE_FIX.md` §2 拟议一致。

### 1.3 锚点并集（§3.2）
$$
\mathcal{A}_*(t) = \mathcal{C}_*(t) \cup \mathcal{R}_*(t),\quad *\in\{u,v\}
$$
（$\mathcal{C}_*$ = common-neighbor-aware nodes）合称 **sampling-aware nodes**，作为采样锚点。

### 1.4 采样流程（§3.2 "Sample-Aware Node Sampling"，论文 sec:sampling）
对每个锚点 $a \in \mathcal{A}_*(t)$（其时刻 $t_a$）：
$$
\mathcal{N}_k(*, a, t_a) = \operatorname{Trunc}\Bigl(\bigl[\,(w,t')\mid (*,w,t')\in\mathcal{E},\ t_{\text{prev}}^*(a)<t'<t_a\,\bigr],\ k\Bigr) \oplus (a, t_a)
$$
$$
\mathcal{H}_*(t) = \operatorname{PadTop}\Bigl(\operatorname{Concat}_{a\in\mathcal{A}_*(t)}\bigl[\mathcal{N}_k(*,a,t_a)\bigr],\ N\Bigr)
$$
其中 $t_{\text{prev}}^*(a)=\max\{t'\mid (*,w,t')\in\mathcal{E},\ w\in\mathcal{A}_*(t),\ t'<t_a\}$（无则 $-\infty$）。
Notation：$\operatorname{Trunc}$ 保留最近 $k$ 个；$\operatorname{PadTop}$ 取最近 $N$ 个并零填充；$\operatorname{Concat}$ 按时序组装；$\oplus$ 在窗末**追加锚点本身** $(a,t_a)$。
配套文字（原文）："for each sampling-aware node, we sample $k$ preceding neighbors. If another sampling-aware node is encountered within this window, we sample all neighbors between them. After constructing the sampling sequence, we then fetch the $N$ most recent neighbors."

⇒ **论文机制 = 锚点扩充式采样（= `DESIGN_RAS_RAE_FIX.md` 机制 A）**：R 节点作为锚点，其前 $k$ 邻居上下文窗 + 锚点本身进入序列。**论文不存在"独立直接历史特征流"（机制 B）**。

### 1.5 RAS 关闭的论文语义（供 E-2 掩码对齐）
RAS 关 ⇒ $\mathcal{A}_*(t)$ 不含 $\mathcal{R}_*(t)$，仅 $\mathcal{C}_*(t)$ 作锚点；序列构造其余不变。

## 2. RAE（编码侧）权威定义 — §3.3.2 BTE 内 "Direct evidence"

### 2.1 定位（原文）
"Beyond third-party signals, the historical interactions between $u$ and $v$ themselves provide a direct and often strong prior: the past sign of $(u,v)$ tends to persist in future interactions, reflecting **sign consistency**."

### 2.2 逐位置定义
对 $\mathcal{H}_u(t)$（resp. $\mathcal{H}_v(t)$）第 $i$ 个邻居 $x_i$：若 $x_i=v$（resp. $x_i=u$），则该位置存在历史有符号交互 $(u,v,t_i)$（resp. $(v,u,t_i)$）；其符号构成**一单位**正/负证据：
$$
r_{(u,v),i}^t =
\begin{cases}
[\,1,0\,]^{\top} & \text{若 } (x_i=v\ \text{in}\ \mathcal{H}_u(t))\lor(x_i=u\ \text{in}\ \mathcal{H}_v(t)) \land \text{sgn}=+1,\\[2pt]
[\,0,1\,]^{\top} & \text{若同条件且 } \text{sgn}=-1,\\[2pt]
[\,0,0\,]^{\top} & \text{否则}.
\end{cases}
$$

### 2.3 与 indirect 合并（"Unified effect encoding"）
$$
b_{(u,v),i}^t =
\underbrace{\begin{cases}
[\,\mathrm{count}(+1\in N_{(u,v),x_i}^t),\ \mathrm{count}(-1\in N_{(u,v),x_i}^t)\,]^{\top} & x_i\in\mathcal{CS}_{(u,v)}^t,\\
[\,0,0\,]^{\top} & \text{否则}
\end{cases}}_{\text{balance-theoretic (indirect)}}
\;+\;
\underbrace{r_{(u,v),i}^t}_{\text{repeat-aware (direct)}}
$$
$B_*^t\in\mathbb{R}^{|\mathcal{H}_*(t)|\times 2}$ 逐位置堆叠；再经 $Z_{*,B}^t = g(B_*^t)$（$g$ = per-row 两层 MLP，首层 $\mathrm{Linear}(2\to d_C)$，Eq.13 已按方案 A 修订为联合投影）。**RAE 证据与 indirect 共用同一 $g$，论文未给 RAE 独立参数**（与代码"复用 BTE 共享层、无新增参数"一致）。

### 2.4 论文明确注释
"a counterpart node is never a common neighbor (no self-loops exist), hence indirect and direct evidence originate from disjoint positions in $\mathcal{H}_*(t)$."
⇒ 前提：**无自环**；direct 证据只出现在"对方节点被采入本侧序列并保留（前 $N$ 内）"的位置。若 $v$ 未被采入 $\mathcal{H}_u$（重复对过旧/被截断），该位置无 direct 证据 ⇒ **RAE 触发依赖 RAS 是否把 $v$ 采入序列**。

### 2.5 RAE 关闭的论文语义
RAE 关 ⇒ $b_i$ 只含 indirect 项（仅统计 $\mathcal{CS}$ 位置 count）。

## 3. 与代码现状对照（A 侧核对结论，供 B 定 ①/②）
- **RAS**：论文语义 = 当 $v\in\mathcal{N}_u(t)$ 时把该直接历史位置**新增**为锚点 $a$（含其前 $k$ 窗 + 锚点本身）。代码 repeat 分支只"改写已有交集 key"（需自环）→ **与论文不符**（B 审计确认）。`DESIGN_RAS_RAE_FIX.md` 机制 A 的修复方向与论文一致。
- **RAE**：论文语义 = 对已采入序列的 counterpart 位置按其**实际符号**产生 $r_i$（$[1,0]/[0,1]$ 通道证据），与 indirect 相加进 $B$。代码把 $[\mathrm{src},\mathrm{dst}]$ `np.append` 进 common-neighbor 集合，因 padded pos0=自身 使 $u,v$ 已"天然在交集中"而**冗余 no-op**。⚠️ 关键：代码的 indirect/CN 证据路径可能已隐式包含 counterpart 位置 —— 修复时需界定 $r_i$（direct）与 indirect 如何在通道上区分/相加，才能产生**非零增量**。论文的相加式（indirect + direct 同通道累加）意味着：direct 证据与 indirect 在**同一计数通道**累加后统一投影 —— 若 indirect 路径已计入了 counterpart 的符号，则按论文语义 RAE 本就是"并入计数"而非独立增量？**此点需 B 对照代码判明**：论文公式里 direct 是独立分量（不同位置），代码因 pos0 自含把两位置合并 → 对齐方式二选一：(a) 让序列中 counterpart 位置单独计入 direct（indirect 只计真正第三方 $w\notin\{u,v\}$），或 (b) 明确论文即"并入计数、无独立通道"，则 RAE 的"模块开关"只影响是否额外 append（无增量=现状）。**这是 ① 方案能否产生消融增量的核心，建议由 B 先做小实验界定增量空间（`DESIGN` §4.4 开放问题 4）**。
- **论文 RAE 不含**"重复次数/最近符号/一致性"聚合强特征（`DESIGN` §2 拟议强于论文）。若走①且严格按论文，RAE 仅为 per-position 一单位符号证据（无新通道/新参数）；做强特征 = 超论文范围，需同步改论文或另议（C5 决策点）。

## 4. 开放问题答复（对应 `DESIGN_RAS_RAE_FIX.md` §4）
| # | 问题 | A 侧答复（基于论文） |
|---|---|---|
| 2 | 论文把 RAS/RAE 描述成什么机制？ | **RAS = 采样侧锚点扩充（机制 A）**；无独立特征流（机制 B 论文未提）。**RAE = BTE 内 per-position direct 分量**（与 indirect 相加进 $B$，共享 $g$，无独立参数）。 |
| 3 | 是否接受新增通道/参数？ | 论文不新增（RAE 走 $B$ 的 2 通道 + 共享 $g$）。若实现需要新通道即超论文 → 须改论文，属 C5 范围。 |
| 4 | 消融能否展示增量？ | 论文语义下 RAE 增量取决于代码是否已隐式含 counterpart 位置证据（见 §3 RAE 段）；RAS 增量应体现在 $v\in\mathcal{N}_u$ 重复对占比高的数据集（Bitcoin ~40%）。建议 B 先量化界定。 |

## 5. 仍待导师/用户定（C5，A 不代决）
- ① 按论文修代码（RAS 补锚点 + RAE 位置符号证据，建议按论文**最小语义**实现、不加次数/一致性特征）→ 重跑 E-2/主表；或
- ② 改论文表述对齐现状（full ≡ BTE+CNAS，消融改 BTE/CNAS 维度）。
- 主表重跑维持暂缓，直至 C5 定案。
