# 修复计划：RAS + RAE + BTE pos0 标签泄漏（一次实现、一次重跑）

> 2026-09-09 起草（Agent B）｜依据：`docs/PAPER_RAS_RAE_SPEC.md`（论文权威语义）+ 泄漏实测证据。
> 状态：⬜ 待用户批准后实现。

## 0. 背景与目标

三个问题**一起修**（相互纠缠，拆开会造成中间状态不可用）：
1. **BTE pos0 标签泄漏**（已实测确认）：`padded_nodes_neighbor_sign[pos0]=当前边标签`，经共同邻居配对 `suggest=标签×历史符号` 泄进特征 → 重复边指标虚高。
2. **RAE no-op**：`np.append([u,v])` 结构上无效果；应改为论文 §3.3.2 的 per-position direct 证据（历史符号，非标签）。
3. **RAS 语义错位**：repeat 分支只在自环时改写已有键；应改为论文 §3.2 的 R 锚点扩充（`v∈N(u)` 即新增锚点）。

修复后语义与论文对齐（indirect=真第三方 only；RAE=counterpart 历史符号证据；RAS=直接历史锚点采样），且**无泄漏**。

---

## 1. 改动 1 — 封泄漏 + indirect 只收真第三方（`models/NeighborInteractEncoder.py`）

**现状**：`count_neighbor_sign_effect`（~246）对含 pos0 的整条 padded 序列 `np.unique → intersect1d`（283-291）→ `sign_effect_count`（103）用 `np.isin` 选两侧所有出现位置（含 pos0）配对（169-181）→ `suggest=src_sign×dst_sign`。

**改动**：
- 在 `count_neighbor_sign_effect` 每对节点内，**只用历史切片**参与：`src_hist = src_padded_node_neighbor_ids[1:]`（丢 pos0），`dst_hist` 同理；sign 数组同步切片。
- `common_neighbor = intersect1d(unique(src_hist), unique(dst_hist))` → 自动只含**真第三方**（u/v 各自只在自己侧 pos0，去掉后互不在对方历史交集；见推论）。
- **删除** RAE 的 `np.append(common_neighbor,[src_id,dst_id])`（293-297，不再需要）。
- `sign_effect_count` 收到的是纯历史数组 → pos0 标签不再进入任何配对。

**效果断言**（本地单测）：Case2/Case3（重复对）翻转 pos0 标签 → 输出 **identical=True**（泄漏消除）；真第三方证据保留（Case1 不变）。

---

## 2. 改动 2 — RAE direct 证据（同文件 `InteractSignEffect` 分支）

**现状**：无 direct 概念；BTE 特征只来自 indirect 计数，散回各位置。

**改动**（在 `count_neighbor_sign_effect` 或 forward 内，`module_repeat_aware_sign_encoder` 门控）：
- 对 `seq(u)` 历史位置 i≥1：若 `neighbor_id[i]==v`（dst_id），该位置 r_i = [1,0]（历史 sign=+1）或 [0,1]（sign=-1）；否则 [0,0]。
- 对 `seq(v)` 对称：`neighbor_id[i]==u`（src_id）→ 按历史 sign 给 [1,0]/[0,1]。
- 组合：`B = indirect_counts + direct`（同一 [pos,neg] 2 通道），再进现有 `neighbor_sign_effect_layer`（共享层，无新参数）。
- RAE 关 ⇒ direct 项为 0 ⇒ 行为 = 改动 1 后的纯 indirect（真第三方）。

**效果断言**：重复场景下 RAE on ≠ RAE off（非零增量空间）；全新对二者相同。

---

## 3. 改动 3 — RAS 锚点扩充（`utils/direct_neighbor_sampler.py`）

**现状**：`common_neighbor_location`（76-112）repeat 分支只 `aware_nodes.get(dst)` 改写**已有键**（需自环）。

**改动**：
- repeat_aware=True 时**无条件新增**：
  - 若 `v`(dst) 在 src 历史中（u 与 v 直接交互过）→ `aware_nodes[v] = (v 在 src 历史的位置, [])`（若已存在则并入 src 位置）；
  - 若 `u`(src) 在 dst 历史中 → `aware_nodes[u] = ([], u 在 dst 历史的位置)`；
- `look_forward_sampling`（732-790）天然支持单侧锚点（src_pos 只生成 src 窗、dst_pos 只生成 dst 窗）→ 无需改动。
- 若无任何 common 且无 R 锚点 → 维持现有 fallback。

**效果断言**：RAS 触发边占比 ≈ 重复边占比（Bitcoin ~40%）；有直接历史的 (u,v) 采样序列 ≠ 关闭态。

---

## 4. 改动 4 — pos0 sign 置中性（`models/SignDyGFormer.py:534`）

- `padded_nodes_neighbor_sign[idx,0] = node_interact_sign[idx]` → 置 **0**（pos0 无历史符号语义）。
- 前提核对：padded sign 数组唯一消费方是编码器 sign-effect 计数（改动 1 后不再含 pos0）；node/edge/time 特征不用它。置 0 双保险防再犯。
- `node_interact_sign` 参数仍传入（历史邻居 sign 需要），仅不再写进 pos0。

---

## 5. 验证计划

| 阶段 | 内容 | 位置/成本 |
|---|---|---|
| 1 单测 | 改动 1/2/3 断言（上述"效果断言"） | 本地 CPU，分钟级 |
| 2 触发率 | RAS 触发占比 vs 重复率；RAE 非零位置占比 | 本地脚本 |
| 3 GPU 小验证 | BitcoinAlpha linksign seed42 单 run：修复前 vs 修复后（保留当前结果文件对照）；重复边子集/整体指标变化 | 服务器 GPU ~1h |
| 4 通过后全量 | E-2(20) / E-3(8) / E-4(2) / E-5(10) / 主表（含基线）双卡 GPU 重跑 | ~1 周 |

**预期**：指标可能回落（去虚高）；是否仍超基线以实测为准；若 RAE 增量/主结论成立则走 ①（论文保留 RAS/RAE 表述，对齐新语义），否则与 A/导师议 ②。

---

## 6. 风险与依赖

- 指标回落幅度不确定 → 需导师知情（核心有效性修复，属诚实做法）。
- 主表/消融/E-3/E-4/E-5 全部重跑（E-3 已在论文 §4.5 的数字作废待换）。
- Agent A 需同步 §3.2/§3.3 表述与消融表（若走①，代码=论文语义，表述不变即可）。
- DirectSignDyGFormer 走同一 BTE 代码，需一并验证（本次计划默认含）。

## 7. 门禁

1. 用户批准本计划 → 2. 实现改动 1-4 → 3. 本地单测+断言全绿 → 4. GPU 小验证（阶段 3）→ 5. 结果回报用户/导师 → 6. 全量重跑。
