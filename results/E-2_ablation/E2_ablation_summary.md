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
6. ⚠️ **BTE 证据另备**：本表四组均 BTE=on；「BTE 主贡献」的修复后证据 = **ⓑ BTE-off 对照**（CNAS-only `[F,F,F,T]` + vanilla `[F,F,F,F]` × 5 数据集；2026-09-13 晚队列 #18）——齐后并入本文件。
7. 交叉验证：RT full 0.9386 = E-5 seed42；RB full 0.9239 = E-5 seed42 / E-3 P1 独立复跑——**逐字一致** ✓。

## 归档与证据
- 原始 JSON（20 个，v3）：`results/E-2_ablation/raw/{Dataset}/`（**gitignored**）。
- 同步记录：`results/_sync_raw_log.csv`（**14:27:07 批次，20 行 sha256**）。
- 完成时间：2026-09-13 14:26（服务器；队列任务 #14/#15；双卡）。

