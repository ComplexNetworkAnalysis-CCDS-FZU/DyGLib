"""c1c2_table.py — DySDGNN 可见性对照 C0/C1/C2 对照表（2026-10-01）。

输入：results/c1c2/raw/{ds}_seed{s}_C012.json（Baseline 648c3fe 一次出三档）
      results/baseline_m5/raw/DySDGNN/{ds}_seed{s}.json（既有正式主表，C0 对账锚）
输出：results/c1c2_table_20261001.txt（含 C0 位级闸门 + 三档 mean±std + Δ‰ + 逐种子）
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
NEW = ROOT / "results" / "c1c2" / "raw"
OLD = ROOT / "results" / "baseline_m5" / "raw" / "DySDGNN"
OUT = ROOT / "results" / "c1c2_table_20261001.txt"

DS = ["BitcoinAlpha", "BitcoinOTC", "WikiVote"]
SEEDS = [42, 123, 456, 789, 1024]
TIERS = ["C0", "C1", "C2"]

lines: list[str] = []
w = lines.append


def fmt(x: float) -> str:
    return f"{x:.4f}"


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


w("DySDGNN 严格可见性对照：C0/C1/C2 三档对照表（Code 独立复核 · 2026-10-01）")
w("数据：results/c1c2/raw/{ds}_seed{s}_C012.json（Baseline 648c3fe，--eval-protocol ALL，同 run 同 ckpt）")
w("口径：C0=转导锚（目标边参与自身嵌入）；C1=逐边精确 mask-self；C2=严格过去（motif 全掩蔽 + MT-SA 剔除 k）")
w("")
w("=" * 96)
w("【闸门 1】同 run 内 metrics(主口径) 与 protocol_metrics.C0 必须逐位一致")
w("=" * 96)
gate1_bad = []
for ds in DS:
    for s in SEEDS:
        d = load(NEW / f"{ds}_seed{s}_C012.json")
        m = d["metrics"]
        c0 = d["protocol_metrics"]["C0"]
        for k in ("AUC", "F1_bin"):
            if m[k] != c0[k]:
                gate1_bad.append(f"{ds}/seed{s}/{k}: metrics={m[k]!r} != C0={c0[k]!r}")
w(f"  15 run × 2 指标 全等 = {not gate1_bad}" + ("" if not gate1_bad else f"；不一致项：{gate1_bad}"))
w("")
w("=" * 96)
w("【闸门 2】C0 与既有正式主表（results/baseline_m5/raw/DySDGNN/）逐位一致（= 本次改动零回归）")
w("=" * 96)
gate2_bad = []
rows_old = {}
for ds in DS:
    for s in SEEDS:
        o = load(OLD / f"{ds}_seed{s}.json")
        n0 = load(NEW / f"{ds}_seed{s}_C012.json")["protocol_metrics"]["C0"]
        rows_old[(ds, s)] = o
        for k in ("AUC", "F1_bin"):
            a, b = o["metrics"][k], n0[k]
            if a != b:
                gate2_bad.append(f"{ds}/seed{s}/{k}: 主表={a!r} vs C0={b!r} (Δ={b - a:+.3e})")
w(f"  15 run × 2 指标 全等 = {not gate2_bad}")
for x in gate2_bad:
    w(f"    ! {x}")
w("")
w("=" * 96)
w("【主表】AUC：C0 / C1 / C2（5 种子 mean±std；Δ‰ = 相对 C0）")
w("=" * 96)
hdr = f"{'数据集':<14}{'C0':>18}{'C1':>18}{'C2':>18}{'ΔC1‰':>9}{'ΔC2‰':>9}"
w(hdr)
w("-" * 96)
summary: dict[str, dict[str, float]] = {}
for ds in DS:
    vals = {t: [load(NEW / f"{ds}_seed{s}_C012.json")["protocol_metrics"][t]["AUC"] for s in SEEDS]
            for t in TIERS}
    mean = {t: st.mean(vals[t]) for t in TIERS}
    std = {t: (st.stdev(vals[t]) if len(vals[t]) > 1 else 0.0) for t in TIERS}
    summary[ds] = dict(mean=mean, std=std, vals=vals)
    d1 = (mean["C1"] - mean["C0"]) * 1000
    d2 = (mean["C2"] - mean["C0"]) * 1000
    w(f"{ds:<14}{mean['C0']:>10.4f}±{std['C0']:.4f}{mean['C1']:>10.4f}±{std['C1']:.4f}"
      f"{mean['C2']:>10.4f}±{std['C2']:.4f}{d1:>+9.1f}{d2:>+9.1f}")
w("")
w("=" * 96)
w("【逐种子 AUC】")
w("=" * 96)
for ds in DS:
    w(f"-- {ds}")
    for s in SEEDS:
        pm = load(NEW / f"{ds}_seed{s}_C012.json")["protocol_metrics"]
        w(f"   seed{s:<6}" + "  ".join(f"{t}={pm[t]['AUC']:.6f}" for t in TIERS)
          + f"   (C1−C0={((pm['C1']['AUC'] - pm['C0']['AUC']) * 1000):+.1f}‰,"
            f" C2−C0={((pm['C2']['AUC'] - pm['C0']['AUC']) * 1000):+.1f}‰)")
w("")
w("=" * 96)
w("【F1_bin：三档是否同样退化】")
w("=" * 96)
for ds in DS:
    s0 = SEEDS[0]
    pm = load(NEW / f"{ds}_seed{s0}_C012.json")["protocol_metrics"]
    same_all = all(
        load(NEW / f"{ds}_seed{s}_C012.json")["protocol_metrics"]["C0"]["F1_bin"]
        == load(NEW / f"{ds}_seed{s}_C012.json")["protocol_metrics"][t]["F1_bin"]
        for s in SEEDS for t in TIERS
    )
    w(f"  {ds:<14} C0/C1/C2 F1_bin = {pm['C0']['F1_bin']:.6f} / {pm['C1']['F1_bin']:.6f} / "
      f"{pm['C2']['F1_bin']:.6f} ；全 15 run 跨档恒等 = {same_all}")
w("")
w("=" * 96)
w("【跨档结论变化判读（供 Paper 直接引用）】")
w("=" * 96)
for ds in DS:
    m = summary[ds]["mean"]
    d1 = (m["C1"] - m["C0"]) * 1000
    d2 = (m["C2"] - m["C0"]) * 1000
    lvl0, lvl1, lvl2 = m["C0"], m["C1"], m["C2"]
    verdict = ("C1≈C0（逐边 mask-self 几乎不改结论）" if abs(d1) < 5 else f"C1 相对 C0 变化 {d1:+.1f}‰")
    verdict2 = ("C2 显著低于 C0" if d2 < -20 else f"C2 相对 C0 {d2:+.1f}‰")
    w(f"  {ds:<14} AUC {lvl0:.4f} → {lvl1:.4f} → {lvl2:.4f}：{verdict}；{verdict2}"
      f"（严格过去下从 {lvl0:.3f} 掉到 {lvl2:.3f}）")
w("")
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"[ok] {OUT.relative_to(ROOT)}")
