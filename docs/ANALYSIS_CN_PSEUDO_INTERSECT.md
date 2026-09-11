# 分析：CN 采样中的伪交集怪癖（assume_unique）——代码实现 vs 论文定义

> 日期：2026-09-11 ｜ 来源：`Perf` 实证（HANDOFF 09-11）+ `Code` 定位与量化（本分析）
> 状态：**待用户/导师定**（保持 vs 修复 vs 先 A/B）。本分析只定性/定量，**不改代码**。

## 1. 代码定位
- `utils/direct_neighbor_sampler.py:86`：
  ```python
  common_vals = np.intersect1d(src_neighbor, dst_neighbor, assume_unique=True)
  ```
  （`common_neighbor_location`；被**无向** `get_common_neighbors`（生产主路径，SignDyGFormer 使用）与**有向** `get_directed_common_neighbors`（Direct 模型）共用；**全库唯一一处 `assume_unique`**。）
- 输入是**原始历史切片**（含重复 id —— 重复边每发生一次就追加一条）。

## 2. numpy 实际语义（numpy 1.26.4 实测）
`assume_unique=True` 的实现 = 排序拼接后取"与前一个元素相等"的所有元素：
- 值 v（左多重数 c1、右多重数 c2）被输出 ⇔ **c1 + c2 ≥ 2**；输出 k−1 份（k=c1+c2；下游为 dict 键，去重后无影响）。
- 玩具探针：`[1,1] vs [] → [1]`（伪交集！）、`[1,1] vs [1] → [1,1]`、`[1,2] vs [2,3] → [2]`。
- 形式化：**C_impl = { w : mult_u(w) + mult_v(w) ≥ 2 }** —— 含"**单侧重复 ≥2、另一侧 0 次**"的**伪共同邻居**。

## 3. 论文定义（`docs/PAPER_RAS_RAE_SPEC.md` §3.2 摘录）
- 采样锚点 A_*(t) = C_*(t) ∪ R_*(t)；C_* = common-neighbor-aware nodes；全节为**集合**记号（N_k、H_* 等）。
- 集合语义下 **C = N_u(t) ∩ N_v(t)**（真交集：两侧都出现、无重复概念）。
  > 注：A 摘录未逐字展开 C 的公式；按 §3.2 集合语义与 CNAS 命名即上述含义（如需逐字引用请 `Paper` 补一行）。
- **差异刻画：C_impl = C ∪ S**，其中 S = {w: 单侧重复≥2 且另一侧 0} \ C。
  - S 中 w∈{u,v}（**对方节点重复**）的部分 → 与 R_* 语义**重叠**（R 本就纳入对方全部历史位置；RAS 修复后实现亦新增之）。
  - **纯偏差 = S \ {u,v}**：与配对无关的单侧重复邻居 —— 论文 A = C ∪ R 不包含它们。

## 4. 真实数据量化（2000 查询/数据集；linksign 生产 k；numpy 1.26.4）

| 数据集 | 伪 CN 查询% | 均伪 CN/查询 | 含 u/v% | 采样输出差异% | 均Δ选中位置 |
|---|---|---|---|---|---|
| BitcoinAlpha | 96.5% | 42.4 | 0.0% | 62.0% | +35.0 |
| BitcoinOTC | 96.8% | 50.5 | 0.0% | 79.2% | +56.9 |
| RedditBody | 66.1% | 4.4 | 21.1% | 52.7% | +5.8 |
| RedditTitle | 71.0% | 10.8 | 11.7% | 69.3% | +12.7 |
| WikiVote | 51.6% | 3.1 | 1.9% | 48.5% | +8.4 |

- "采样输出差异" = **只替换 intersect 这一处**（其余 `look_forward_sampling` 等逐行同生产实现、RA=off）时选中位置集合的差异；为**窗口选择层**的差异，后续 PadTop(N) 截断会削弱其传导到 H 序列的程度，**最终指标影响以训练 A/B 实测为准**。
- 合成图交叉印证：Perf 侧 908 查询中 822 受影响（90.5%）、65% 输出差异 —— 与本表同量级。

## 5. 影响链与现状
- 额外锚点 → look_forward 窗口（含既有锚点的窗口边界也会变化——**非简单超集**）→ H_* 序列 → 模型。
- **BTE/RAE 不受影响**（K2 先 `unique` 后 intersect，真交集）。
- **全部现存结果（E-2/E-3/E-4/E-5、旧主表）均在 C_impl 语义下产出** —— 这是"实际运行的真相"；差异集中在重复率高的数据集（BA/OTC，与 RAS 作用面相同）。

## 6. 选项（待定）
1. **保持**（成本 0）：以 C_impl 为实现语义；若论文严格声明 C = 共同邻居，存在表述与实现的偏差（= 单侧重复邻居），可在修订说明/回复信中如实披露。
2. **修复**（一行改动：改真交集）：语义对齐论文；但采样分布改变（上表 48–79%）→ **所有结果需再次重跑**（E-2/E-3/E-4/E-5/主表），成本大。
3. **先 A/B 量化**（推荐）：同配置 seed42，quirk vs fix，先在代表性数据集（如 BitcoinAlpha + RedditTitle）各 2 run；若指标差 ≲0.002 且方向稳定 → "保持 + 如实说明"有据；若显著 → 修复 + 全量重跑。成本：4 run ≈ 半日（双卡）。

## 7. 证据文件
- 本次：`tools/verify/cn_quirk_analysis.py`（脚本）+ `tools/verify/cn_quirk_stats.json`（本地指标，json 按 gitignore 不入库）
- Perf 侧：`D:\codes\SignDyG-Perf\tools\probe_numpy_quirks.py`、`tools\quirk_impact_k1.py`
