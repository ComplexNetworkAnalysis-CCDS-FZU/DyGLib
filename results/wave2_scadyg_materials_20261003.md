# ScaDyG 修复后 15/15 材料（Code 2026-10-03）

## 逐件 sha256（fetch 入账 `results/_sync_raw_log.csv`；remote = 服务器 `repro/outputs/ScaDyG/`）

- `94f3739520b7797e`  BitcoinAlpha_seed1024.json  (1440 B)
- `b862f51c12d95bd3`  BitcoinAlpha_seed123.json  (1438 B)
- `83a9a68efc4a7272`  BitcoinAlpha_seed42.json  (1439 B)
- `42bcb5d4009505c1`  BitcoinAlpha_seed456.json  (1440 B)
- `72ac058b0c765188`  BitcoinAlpha_seed789.json  (1439 B)
- `3ac0f314e7a455ac`  BitcoinOTC_seed1024.json  (1439 B)
- `84209eb42664788f`  BitcoinOTC_seed123.json  (1435 B)
- `b0c756459a7d710f`  BitcoinOTC_seed42.json  (1435 B)
- `bd7f18f15a8c46a8`  BitcoinOTC_seed456.json  (1438 B)
- `ab52283c5c34f8c5`  BitcoinOTC_seed789.json  (1436 B)
- `804afdcfd485fab6`  WikiVote_seed1024.json  (1433 B)
- `12fc90e43537be9b`  WikiVote_seed123.json  (1434 B)
- `91cb3e2a4c9448ea`  WikiVote_seed42.json  (1432 B)
- `1acbd2c6b61b8316`  WikiVote_seed456.json  (1430 B)
- `313ca89373459478`  WikiVote_seed789.json  (1435 B)

## 边序对齐（`[align]` 行，来自 `tools/queue/logs/task_738.log`）
- BA / OTC / WV 三数据集均为 `逐片位移边数=[0, 0, ..., 0]`（对应各数据集分片数 15/25/15 全 0）
  ⇒ 本次转换的边序与 DGL 读取序一致（无错配）。

## 数值表
- `results/wave2_scadyg_table_20261002.txt`（mean±std 用 **ddof=1**，与主表一致）
- 与 0.5 的逐点差（‰）：BA +16.0 +71.3 +57.7 +63.8 +70.2；OTC +255.6 +263.3 +239.4 +237.9 +262.4；WV +52.6 +31.4 +37.1 +47.0 +21.1
- val_AUC 均值：BA .6874 / OTC .6503 / WV .5900（旧档 09-28 恒 0.5000）

## 复核要点（Baseline 侧）
- `val_AUC ≠ 0.5` ✓、`auc ≠ 0.5` ✓、std>0 ✓、非固定偏移 ✓ ⇒ 满足 Paper `e755 §三` 入表条件。
- F1_bin：BA 仍常数 .9417（多数类退化）；OTC/WV 随种子变化 ⇒ 表只列 AUC。
