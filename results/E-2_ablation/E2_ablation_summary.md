# E-2 消融（修复后代码 · 全量 20/20）

> 协议：**linksign**（link&sign 3 类）；5 数据集；seed 42；消融定稿超参 NN-Best·LF-Best；**P1**、**TE**；**GPU（cuda:0/1）**；代码 = 泄漏/RAS/RAE 修复后（2026-09-09 fix，部署校验通过）。
> 配置位：base = `RAS-D·RASE-D`；+RAE = `RASE-E`；+RAS = `RAS-E`；full = `RAS-E·RASE-E`；`BTE-E·CNAS-E` 四组恒开。
> 完成：2026-09-11 17:01（双卡并行，20/20）。

## AUC（主指标）

| 数据集 | base | +RAE | +RAS | **full** | 旧代码（含泄漏，全配置同值） |
|---|---|---|---|---|---|
| BitcoinAlpha | 0.9524 | 0.9532 | **0.9606** | 0.9554 | 0.9649 |
| BitcoinOTC | 0.9676 | 0.9680 | **0.9746** | 0.9720 | — |
| WikiVote | 0.9620 | 0.9630 | 0.9628 | 0.9629 | — |
| RedditTitle | 0.9337 | 0.9348 | 0.9366 | **0.9373** | 0.9539 |
| RedditBody | 0.9265 | 0.9249 | 0.9282 | **0.9299** | 0.9610 |

## 明细（AUC / AP / sign-F1 / 单 run 耗时）

| 数据集 | 配置 | AUC | AP | sign-F1 | t(s) |
|---|---|---|---|---|---|
| BitcoinAlpha | base | 0.9524 | 0.7926 | 0.9037 | 4779 |
| BitcoinAlpha | +RAE | 0.9532 | 0.7954 | 0.8924 | 4920 |
| BitcoinAlpha | +RAS | 0.9606 | 0.8103 | 0.9230 | 6560 |
| BitcoinAlpha | full | 0.9554 | 0.8006 | 0.9074 | 4953 |
| BitcoinOTC | base | 0.9676 | 0.8345 | 0.9189 | 11025 |
| BitcoinOTC | +RAE | 0.9680 | 0.8361 | 0.9286 | 11461 |
| BitcoinOTC | +RAS | 0.9746 | 0.8544 | 0.9338 | 12921 |
| BitcoinOTC | full | 0.9720 | 0.8493 | 0.9301 | 11028 |
| WikiVote | base | 0.9620 | 0.8160 | 0.8707 | 3732 |
| WikiVote | +RAE | 0.9630 | 0.8180 | 0.8708 | 5158 |
| WikiVote | +RAS | 0.9628 | 0.8165 | 0.8760 | 4748 |
| WikiVote | full | 0.9629 | 0.8188 | 0.8742 | 4562 |
| RedditTitle | base | 0.9337 | 0.7070 | 0.7961 | 3508 |
| RedditTitle | +RAE | 0.9348 | 0.7070 | 0.8151 | 5844 |
| RedditTitle | +RAS | 0.9366 | 0.7111 | 0.8459 | 7673 |
| RedditTitle | full | 0.9373 | 0.7152 | 0.8222 | 6963 |
| RedditBody | base | 0.9265 | 0.6926 | 0.9459 | 4097 |
| RedditBody | +RAE | 0.9249 | 0.6934 | 0.9325 | 5024 |
| RedditBody | +RAS | 0.9282 | 0.6940 | 0.9421 | 4132 |
| RedditBody | full | 0.9299 | 0.6948 | 0.9241 | 7559 |

## 结论
1. **RAS 5/5 一致正增益**（+RAS − base）：BA **+0.0082** / OTC **+0.0070** / WV +0.0008 / RT +0.0029 / RB +0.0017。
2. **RAE 单独 ≈ 中性**：−0.0016（RB）~ +0.0011（RT）。
3. **full ≥ base（5/5）**，但增量主要来自 RAS；**Bitcoin 两数据集 full < +RAS**（BA −0.0052、OTC −0.0026），RT/RB 上 full 略优（+0.0007/+0.0017）。
4. 较旧（泄漏）代码全面下降：BA −0.0095 / RT −0.0166 / RB −0.0311 → **泄漏曾系统性高估**。
5. 单种子（42）口径；多种子确认可并入 E-5 扩展（10 种子）。

## 归档与证据
- 原始 JSON（20 个）：`results/E-2_ablation/raw/{Dataset}/`（**gitignored**，仅本地/服务器）。
- 同步记录：`results/_sync_raw_log.csv`（sha256）。
- 交叉验证：RB full 0.9299、BA full 0.9554 与修复后验证 run 一致。
