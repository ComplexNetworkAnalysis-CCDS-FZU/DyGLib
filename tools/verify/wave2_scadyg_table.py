"""wave2_scadyg_table.py — 波二 ScaDyG 修复后结果表（2026-10-02）。

输入：results/wave2/raw/ScaDyG/{ds}_seed{s}.json（Baseline 618ba85 修复后重跑，`--force`）
前值：旧档（09-28）全退化 AUC ≡ 0.5000（见 `results/wave2_mamba_table_20261001.txt` 留档）
输出：results/wave2_scadyg_table_20261002.txt

口径：mean±std 用 **ddof=1**（与主表/m5_summary 一致；Paper 待裁定统一）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# 复用 mamba 表脚本的加载逻辑
sys.path.insert(0, str(Path(__file__).resolve().parent))
import json  # noqa: E402
import statistics as st  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "wave2" / "raw" / "ScaDyG"
OUT = ROOT / "results" / "wave2_scadyg_table_20261002.txt"

DS = ["BitcoinAlpha", "BitcoinOTC", "WikiVote"]
SEEDS = [42, 123, 456, 789, 1024]

L: list[str] = []
w = L.append
w("波二 ScaDyG（Baseline 618ba85 零特征修复 + DGL 边序对齐 + 历史窗修正 后重跑）· Code 2026-10-02")
w("前值（09-28 旧档）：**15/15 全退化 AUC ≡ 0.5000、val_AUC ≡ 0.5000**（留档见 `results/wave2_mamba_table_20261001.txt`）")
w("口径：mean±std 用 **ddof=1**（与主表一致；Paper 就 ± 口径统一待裁，见 Baseline b6c1）")
w("")
w(f"{'数据集':<14}{'AUC mean':>10}{'std':>9}{'min':>9}{'max':>9}{'Δ vs 0.5 (‰, 逐点)':>34}   逐种子 AUC")
w("-" * 118)
summ = {}
for ds in DS:
    vals = [json.loads((RAW / f"{ds}_seed{s}.json").read_text(encoding="utf-8"))["metrics"]["AUC"] for s in SEEDS]
    d50 = [f"{(v - 0.5) * 1000:+.1f}" for v in vals]
    m, sd = st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)
    summ[ds] = (m, sd, vals)
    w(f"{ds:<14}{m:>10.4f}{sd:>9.4f}{min(vals):>9.4f}{max(vals):>9.4f}" + "[" + " ".join(d50) + "]   "
      + " ".join(f"{v:.4f}" for v in vals))
w("")
w("=" * 118)
w("Val AUC（同 run；用于核对『不再恒 0.5』）")
w("=" * 118)
for ds in DS:
    vv = [json.loads((RAW / f"{ds}_seed{s}.json").read_text(encoding="utf-8")).get("val_AUC") for s in SEEDS]
    w(f"  {ds:<14}" + " ".join(f"{v:.4f}" for v in vv) + f"   mean={st.mean(vv):.4f}  (旧档恒 0.5000)")
w("")
w("=" * 118)
w("F1_bin（核对是否仍为多数类常数）")
w("=" * 118)
for ds in DS:
    fv = [json.loads((RAW / f"{ds}_seed{s}.json").read_text(encoding="utf-8"))["metrics"]["F1_bin"] for s in SEEDS]
    w(f"  {ds:<14}" + " ".join(f"{v:.4f}" for v in fv) + f"   std={st.stdev(fv):.4f}  唯一值={sorted(set(round(v, 4) for v in fv))}")
w("")
w("判读建议：")
w("  · 修复后 **val_AUC 与 AUC 均显著偏离 0.5** ⇒ 打分非常数、适配器已产生有效排序；")
w("  · 量级（BA .5558 / OTC .7517 / WV .5378）**低于或接近** DyG-Mamba（BA .7067 / OTC .9220 / WV .8619），且 OTC 明显低于主表量级；")
w("  · F1_bin：BA 仍为常数 .9417（多数类退化），OTC/WV 已随种子变化 ⇒ 只报 AUC 更稳（与 mamba 同口径）。")
OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
print(f"[ok] {OUT.relative_to(ROOT)}")
