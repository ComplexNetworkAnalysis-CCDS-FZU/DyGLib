# -*- coding: utf-8 -*-
"""生成 T14 紧凑切片（7 方法 × 2 任务 × 5 数据集 × {auc, 主指标}，mean + n）。"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
rows = list(csv.DictReader(open(ROOT / "results/semba_variant_table_20260924.csv", encoding="utf-8")))
out = []
for r in rows:
    if r["metric"] in ("auc", "main"):
        out.append(f"{r['task']},{r['variant']},{r['dataset']},{r['metric']},{float(r['mean']):.4f},n={r['n']}")
(ROOT / "results/t14_slice.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
print(len(out), "rows")
print("\n".join(out[:6]))
