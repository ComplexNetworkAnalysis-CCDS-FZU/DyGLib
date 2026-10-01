"""paired_stats_recheck.py — 2×2 变体 vs base(Recent-N) 配对统计复核（Code 2026-10-01）。

数据：results/E-2_ablation/raw_seeds/{ds}/SignDyGFormer_seed*.NN-Best.LF-Best.<TAG>.P1.TE.json
     （注意：test metrics 值为 4 位小数字符串 → Δ‰ 精度 ~0.1‰，t/p/d 基于该精度）
输出：results/paired_2x2_recheck_20261001.txt
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "E-2_ablation" / "raw_seeds"
OUT = ROOT / "results" / "paired_2x2_recheck_20261001.txt"

SEEDS = [42, 123, 456, 789, 1024]
DS5 = ["BitcoinAlpha", "BitcoinOTC", "RedditHyperlinkTitle", "RedditHyperlinkBody", "WikiVote"]
DS3 = ["RedditHyperlinkTitle", "RedditHyperlinkBody", "WikiVote"]
DSMAP = {"BitcoinAlpha": "BA", "BitcoinOTC": "OTC", "RedditHyperlinkTitle": "RT",
         "RedditHyperlinkBody": "RB", "WikiVote": "WV"}

CFG = {
    "RAS-E.RASE-E.BTE-E.CNAS-E": "full(RAS+RAE)",
    "RAS-D.RASE-D.BTE-E.CNAS-E": "BTE+CNAS(no RAS/RAE)",
    "RAS-E.RASE-E.BTE-E.CNAS-D": "BTE-only(RAS+RAE)",
    "RAS-E.RASE-E.BTE-D.CNAS-E": "CNAS-only(RAS+RAE)",
    "RAS-D.RASE-D.BTE-E.CNAS-D": "BTE-only(no RAS/RAE)",
    "RAS-D.RASE-D.BTE-D.CNAS-E": "CNAS-only(no RAS/RAE)",
    "RAS-D.RASE-D.BTE-D.CNAS-D": "base/Recent-N(all off)",
}
BASE = "RAS-D.RASE-D.BTE-D.CNAS-D"


def load(ds: str, tag: str, metric: str) -> dict[int, float] | None:
    out = {}
    for p in glob.glob(str(RAW / ds / f"SignDyGFormer_seed*.NN-Best.LF-Best.{tag}.P1.TE.json")):
        seed = int(Path(p).name.split("seed")[1].split(".")[0])
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        out[seed] = float(d["test metrics"].get(metric, "nan"))
    return out or None


def stats(a: dict[int, float], b: dict[int, float]) -> dict:
    seeds = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in seeds]
    n = len(diffs)
    mean = st.mean(diffs)
    sd = st.stdev(diffs) if n > 1 else 0.0
    t = mean / (sd / n ** 0.5) if sd > 0 else float("inf")
    d = mean / sd if sd > 0 else float("inf")
    pos = sum(1 for x in diffs if x > 0)
    try:
        from scipy import stats as ss
        p = float(ss.ttest_rel([a[s] for s in seeds], [b[s] for s in seeds]).pvalue)
    except Exception:  # noqa: BLE001
        p = float("nan")
    return dict(n=n, seeds=seeds, mean=mean, sd=sd, t=t, d=d, pos=pos, p=p, diffs=diffs)


L: list[str] = []
w = L.append
w("2×2 变体 vs base(=Recent-N, all off) 配对统计复核（Code · 2026-10-01）")
w("数据：results/E-2_ablation/raw_seeds/{ds}/…NN-Best.LF-Best.<TAG>.P1.TE.json（**test metrics 为 4 位小数字符串**）")
w("说明：Δ‰ = 变体 − base（同种子配对，n=5）；t = 配对 t；d = mean(diff)/sd(diff)（配对 Cohen's d）。")
w("")

ANCHORS = [  # 98ca §A（linksign auc）用于对齐口径
    ("RT", "full(RAS+RAE)", "auc", +4.4, 4.26, 0.013, 1.91),
    ("RT", "BTE+CNAS(no RAS/RAE)", "auc", -2.1, -3.69, 0.021, -1.65),
    ("RT", "CNAS-only(RAS+RAE)", "auc", -8.7, -7.15, 0.002, -3.20),
    ("RB", "CNAS-only(RAS+RAE)", "auc", -8.8, -3.59, 0.023, -1.61),
    ("WV", "full(RAS+RAE)", "auc", -6.1, -25.32, 0.000, -11.33),
]

for task, ds_list, metrics in (("linksign", DS5, ["auc", "f1_wt", "f1_mac"]),
                               ("sign", DS3, ["f1_macro", "auc", "f1_bin"])):
    w("=" * 100)
    w(f"### {task}")
    w("=" * 100)
    for metric in metrics:
        w(f"--- 指标 {metric}（Δ‰ / 正数 / p / d）")
        w(f"{'数据集':<8}" + "".join(f"{name:>26}" for name in CFG.values() if name != CFG[BASE]))
        for ds in ds_list:
            base = load(ds, BASE, metric)
            if not base:
                w(f"{DSMAP[ds]:<8}  [无 base 文件]")
                continue
            cells = []
            for tag, name in CFG.items():
                if tag == BASE:
                    continue
                var = load(ds, tag, metric)
                if not var:
                    cells.append(f"{'—':>26}")
                    continue
                s = stats(var, base)
                cells.append(f"{s['mean'] * 1000:>+8.1f}/{s['pos']}/5 p={s['p']:.4f} d={s['d']:+.2f}".rjust(26))
            w(f"{DSMAP[ds]:<8}" + "".join(cells))
        w("")
w("=" * 100)
w("【与 98ca §A 锚点对齐检查（linksign auc）】")
w("=" * 100)
for ds, name, metric, dm, tm, pm, dval in ANCHORS:
    base = load(ds, BASE, metric)
    tag = [k for k, v in CFG.items() if v == name][0]
    var = load(ds, tag, metric)
    if not base or not var:
        w(f"  {ds}/{name}: 缺文件")
        continue
    s = stats(var, base)
    ok = abs(s["mean"] * 1000 - dm) < 1.2 and abs(abs(s["t"]) - abs(tm)) < 1.0 and abs(s["d"] - dval) < 0.35
    w(f"  {ds:<4}{name:<24} 复算 Δ={s['mean'] * 1000:+.1f}‰ t={s['t']:+.2f} p={s['p']:.4f} d={s['d']:+.2f}"
      f"   | 98ca: Δ={dm:+.1f}‰ t={tm:+.2f} p={pm:.4f} d={dval:+.2f}  → {'一致' if ok else '不一致'}")
w("")
OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
print(f"[ok] {OUT.relative_to(ROOT)}")
