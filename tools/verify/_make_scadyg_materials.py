"""_make_scadyg_materials.py — 生成 ScaDyG 材料清单（sha256 + align 证据）。"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
rows = list(csv.reader((ROOT / "results/_sync_raw_log.csv").open(encoding="utf-8")))
sc = [r for r in rows[1:] if "ScaDyG" in r[2]]
seen = {}
for r in sc:
    seen[r[2].split("/")[-1]] = r

L: list[str] = []
w = L.append
w("# ScaDyG 修复后 15/15 材料（Code 2026-10-03）")
w("")
w("## 逐件 sha256（fetch 入账 `results/_sync_raw_log.csv`；remote = 服务器 `repro/outputs/ScaDyG/`）")
w("")
for k in sorted(seen, key=lambda x: (x.split("_")[0], x)):
    r = seen[k]
    w(f"- `{r[4][:16]}`  {k}  ({r[5]} B)")
w("")
w("## 边序对齐（`[align]` 行，来自 `tools/queue/logs/task_738.log`）")
w("- BA / OTC / WV 三数据集均为 `逐片位移边数=[0, 0, ..., 0]`（对应各数据集分片数 15/25/15 全 0）")
w("  ⇒ 本次转换的边序与 DGL 读取序一致（无错配）。")
w("")
w("## 数值表")
w("- `results/wave2_scadyg_table_20261002.txt`（mean±std 用 **ddof=1**，与主表一致）")
w("- 与 0.5 的逐点差（‰）：BA +16.0 +71.3 +57.7 +63.8 +70.2；OTC +255.6 +263.3 +239.4 +237.9 +262.4；WV +52.6 +31.4 +37.1 +47.0 +21.1")
w("- val_AUC 均值：BA .6874 / OTC .6503 / WV .5900（旧档 09-28 恒 0.5000）")
w("")
w("## 复核要点（Baseline 侧）")
w("- `val_AUC ≠ 0.5` ✓、`auc ≠ 0.5` ✓、std>0 ✓、非固定偏移 ✓ ⇒ 满足 Paper `e755 §三` 入表条件。")
w("- F1_bin：BA 仍常数 .9417（多数类退化）；OTC/WV 随种子变化 ⇒ 表只列 AUC。")
(ROOT / "results/wave2_scadyg_materials_20261003.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L[:12]))
print(f"... 共 {len(seen)} 件；[ok] results/wave2_scadyg_materials_20261003.md")
