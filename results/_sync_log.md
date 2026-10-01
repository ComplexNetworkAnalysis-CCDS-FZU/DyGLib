# 取数/同步留档（Code 维护）

> 本条为 **Paper 2026-10-01 裁定 §二** 要求的留档条目（服务器工作副本改动）。
> 说明：逐文件 sha256 的机器日志在 `results/_sync_raw_log.csv`（**gitignore**，仅本机留存）；本文件为**可入库**的人工留档。

---

## 2026-10-01 · repro 工作副本「数据根路径」热补丁（H1 路径事故）

| 项 | 内容 |
|---|---|
| 起因 | Baseline 提交 `648c3fe` 把数据根硬编码为 Windows 路径 `D:/codes/DyGLib/processed_data`（`configs/datasets.yaml` 3 处 `csv_path` + `data/snapshot.py:22 DATA_ROOT`）⇒ 服务器 `#723–725`（C1/C2 三行，10-01 03:0x 执行）全部 `FileNotFoundError`。 |
| 位置 | 服务器 `/home/fedsa/DynamiSE_DySDGNN_repro`（**工作副本，未提交**） |
| 改动 | 4 处替换 `D:/codes/DyGLib/processed_data` → `/home/fedsa/DyGLib/processed_data` |
| diff 备份 | `/tmp/repro_prepath_20261001-*.diff`（写入前 `git diff`） |
| 一键还原 | `cd /home/fedsa/DynamiSE_DySDGNN_repro && git checkout -- configs/datasets.yaml data/snapshot.py` |
| 验证 | CPU 冒烟（BA/seed42/2ep）：`protocol_metrics.{C0,C1,C2}` 齐出，C0 AUC 0.6956 = Baseline 自检同值 |
| 上游修复 | Baseline `618ba85`：`DATA_ROOT = env DYSDGNN_DATA_ROOT > 本机默认 > 服务器默认`（存在性自动探测）+ `expand_path()` 支持 `{DATA_ROOT}` 占位；`datasets.yaml`/`datasets_server.yaml` 均改为占位符 ⇒ **服务器零配置直跑** |
| 状态 | 已按 Baseline 指示「先还原补丁 → 再 ff 到 `618ba85`」；本条目保留备查（补丁本身不再需要） |
| 相关 | 本热补丁**只延时**、不改任何数值：C0 与既有正式主表 `results/baseline_m5/raw/DySDGNN/` **15 run × 2 指标逐位一致**（见 `results/c1c2_table_20261001.txt` 闸门 2） |

---

## 2026-09-30 · CNS-FX 队列护栏路径 bug（自伤，仅延时）

- 现象：CNS-FX 行护栏 `[ -f saved_results/<task>/<ds>/... ]` 漏一级 `SignDyGFormer/` ⇒ 条件永假、每行空转 20 min。
- 影响：**仅延时**（结果 JSON 与数值不受影响；FX 复评最终 10/10 与主口径逐位一致 Δ=0.0‰）。
- 处置：`tools/queue/edit_remote_tasks.py --replace-range 720,721`（回读校验：725→725 行不变，备份 `tasks.txt.bak-20260930-213004`）+ 修正生成器 `tools/verify/_gen_cns_fx_rows.py`。
- 教训：护栏路径一律用 `saved_results/<task>/SignDyGFormer/<ds>/`。
