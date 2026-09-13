# E-2 消融（CN 真交集修复版 · v3 · 全量 20/20）

> ✅ **本表为论文可引用版本（v3）**：CN 真交集修复（2026-09-11 晚部署）后重跑（队列 #14/#15），**2026-09-13 14:26 完成（双卡；20/20）**。
> ⛔ 历史代际：v1（含 pos0 泄漏）→ 作废；v2（泄漏/RAE 已修、CN 未修）→ 作废。**旧表述「RAS 5/5 一致正增益」「full≥base（5/5）」作废**——均为伪交集旧语义产物。
> 协议：**linksign**（link&sign 3 类）；5 数据集；**seed 42（单种子）**；消融定稿超参 NN-Best·LF-Best；P1、TE；GPU；加速后端默认启用（bit-exact，JSON 含 `accel` 溯源）。
> 配置位：base = `RAS-D·RASE-D`；+RAE = `RASE-E`；+RAS = `RAS-E`；full = `RAS-E·RASE-E`；`BTE-E·CNAS-E` 四组恒开（**BTE-off 对照见 ⓑ：2026-09-13 晚队列 #18**）。

## AUC（主指标）

| 数据集 | base | +RAE | +RAS | **full** |
|---|---|---|---|---|
| BitcoinAlpha | 0.9554 | 0.9551 | 0.9503 | **0.9586** |
| BitcoinOTC | 0.9684 | 0.9652 | **0.9702** | 0.9668 |
| WikiVote | **0.9608** | 0.9597 | 0.9603 | 0.9607 |
| RedditTitle | 0.9311 | 0.9311 | 0.9354 | **0.9386** |
| RedditBody | **0.9292** | 0.9120 | 0.9266 | 0.9239 |

## 明细（AUC / AP / sign-F1）

| 数据集 | 配置 | AUC | AP | sign-F1 |
|---|---|---|---|---|
| BitcoinAlpha | base | 0.9554 | 0.8056 | 0.9226 |
| BitcoinAlpha | +RAE | 0.9551 | 0.8092 | 0.9240 |
| BitcoinAlpha | +RAS | 0.9503 | 0.7984 | 0.9158 |
| BitcoinAlpha | full | 0.9586 | 0.8195 | 0.9265 |
| BitcoinOTC | base | 0.9684 | 0.8468 | 0.9418 |
| BitcoinOTC | +RAE | 0.9652 | 0.8389 | 0.9360 |
| BitcoinOTC | +RAS | 0.9702 | 0.8528 | 0.9447 |
| BitcoinOTC | full | 0.9668 | 0.8460 | 0.9376 |
| WikiVote | base | 0.9608 | 0.8122 | 0.8613 |
| WikiVote | +RAE | 0.9597 | 0.8129 | 0.8749 |
| WikiVote | +RAS | 0.9603 | 0.8136 | 0.8731 |
| WikiVote | full | 0.9607 | 0.8150 | 0.8732 |
| RedditTitle | base | 0.9311 | 0.7075 | 0.8627 |
| RedditTitle | +RAE | 0.9311 | 0.7080 | 0.8797 |
| RedditTitle | +RAS | 0.9354 | 0.7124 | 0.8732 |
| RedditTitle | full | 0.9386 | 0.7185 | 0.8804 |
| RedditBody | base | 0.9292 | 0.6916 | 0.9395 |
| RedditBody | +RAE | 0.9120 | 0.6752 | 0.9225 |
| RedditBody | +RAS | 0.9266 | 0.6910 | 0.9386 |
| RedditBody | full | 0.9239 | 0.6899 | 0.9328 |

## 结论（v3 · 单种子口径）
1. **组间差多在噪声带内**：对照主表 linksign 的 5 种子 σ（RT 0.0007 / RB 0.0045 / WV 0.0005 / BA 0.0019 / OTC 0.0025），除 RT（+0.0075，≈10σ）外，其余 |Δ| 均 ≲ 0.005。
2. **full vs base**：RT **+0.0075**、BA +0.0032 为正；OTC −0.0016 / WV −0.0001 / RB −0.0053 ≈ 持平（噪声内）。
3. **RAS 单独**：RT **+0.0043**、OTC +0.0018；BA −0.0051 / RB −0.0026 / WV −0.0005 → **不再有「5/5 正增益」**（旧表述作废）。
4. **RAE 单独**：中性偏负（RB **−0.0172** 最明显；BA/WV/RT 在 ±0.001 内）。
5. **最优位**：full 在 BA/RT 最高；OTC 最高 = +RAS；RB 最高 = base——单种子下未能支持「full 全面最优」。
6. ⚠️ **BTE 证据另备**：本表四组均 BTE=on；「BTE 主贡献」的修复后证据 = **ⓑ BTE-off 对照**（CNAS-only `[F,F,F,T]` + vanilla `[F,F,F,F]` × 5 数据集；2026-09-13 晚队列 #18）——**已齐，见下节「ⓑ BTE-off 对照」。**
7. 交叉验证：RT full 0.9386 = E-5 seed42；RB full 0.9239 = E-5 seed42 / E-3 P1 独立复跑——**逐字一致** ✓。

## ⓑ BTE-off 对照（2026-09-13 晚 · 队列 #18 · 单种子 42）

> 补齐 v3 缺口（四组均 BTE=on、无 BTE-off 证据）。本批 = `RAS-D·RASE-D·BTE-D` 下 CNAS-only（`CNAS-E`）与 vanilla（`CNAS-D`）两行 × 5 数据集（10 runs）；协议同 v3（linksign、seed42、P1、TE、GPU、加速默认开）。
> 语义：**CNAS-only = 仅开共邻居采样；vanilla = 全关**；对照列 v3 `base`（BTE+CNAS 全开）。

| 数据集 | CNAS-only（auc/ap/sF1） | vanilla（auc/ap/sF1） | v3 base（BTE 开） | BTE 边际（base−CNAS-only） | CNAS 边际·BTE 关时（CNAS-only−vanilla） |
|---|---|---|---|---|---|
| BitcoinAlpha | 0.9556 / 0.7990 / 0.9096 | 0.9617 / 0.8157 / 0.9263 | 0.9554 | **−0.0002（≈0）** | **−0.0061** |
| BitcoinOTC | 0.9631 / 0.8196 / 0.9330 | 0.9671 / 0.8272 / 0.9357 | 0.9684 | **+0.0053** | −0.0040 |
| WikiVote | 0.9620 / 0.8181 / 0.8695 | 0.9670 / 0.8301 / 0.8853 | 0.9608 | −0.0012 | −0.0050 |
| RedditTitle | 0.9230 / 0.6988 / 0.8534 | 0.9340 / 0.7061 / 0.8871 | 0.9311 | **+0.0081** | **−0.0110** |
| RedditBody | 0.9316 / 0.7002 / 0.9159 | 0.9334 / 0.7005 / 0.9252 | 0.9292 | −0.0024 | −0.0018 |

**结论（单种子，入论文前需多种子坐实）**
1. **BTE 边际**：RT +0.0081 / OTC +0.0053 为正；BA ≈0（−0.0002）；WV −0.0012 / RB −0.0024 微负 → 「BTE 主贡献」需按数据集分述，不能一概而论。
2. **BTE 关时 CNAS 全程为负**（5/5：−0.0018 ~ −0.0110）→ CNAS 收益依赖 BTE 存在（或与 BTE 冗余）。
3. **vanilla ≥ full 于 4/5**（BA +0.0031、OTC +0.0003、WV +0.0063、RB +0.0095；仅 RT full +0.0046）→ 模块整体有效性表述需谨慎，建议对 impactful 组合（RT full、BA/RB vanilla）加种子后定稿。

## 归档与证据
- 原始 JSON（20 个，v3）：`results/E-2_ablation/raw/{Dataset}/`（**gitignored**）。
- 原始 JSON（ⓑ 10 个）：`results/E-2_ablation/raw_bte/{Dataset}/`（**gitignored**）。
- 同步记录：`results/_sync_raw_log.csv`（v3 批次 14:27:07 共 20 行；ⓑ/ⓓ 批次 2026-09-13 晚共 25 行 sha256）。
- 完成时间：2026-09-13 14:26（服务器；队列任务 #14/#15；双卡）。

