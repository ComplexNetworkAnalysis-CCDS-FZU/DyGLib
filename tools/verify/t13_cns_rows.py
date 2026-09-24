# -*- coding: utf-8 -*-
"""T13：两主表 CNAS 行绝对值（linksign: w/o CNAS、CNAS-only；sign: w/o CNAS）。

配对口径：与**同批** full（E-2 raw_seeds `RAS-E.RASE-E.BTE-E.CNAS-E` / sign_wocnas 批的
`CNAS-D` 与 sign full 同协议代）互配，另给 Δ 与 5 种子统计。
输出：results/t13_cns_rows.txt + results/t13_cns_rows.csv
用法：python tools/verify/t13_cns_rows.py
"""
from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DS = ["WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha", "BitcoinOTC"]

LINK_METRICS = ["f1_wt", "f1_mac", "sign_f1", "exist_f1", "f1_mic", "ap", "auc"]
SIGN_METRICS = ["auc", "f1_mac", "f1_binary", "acc", "f1_weighted"]


def _seed(p):
    return int(re.search(r"seed(\d+)", Path(p).name).group(1))


def load(pattern, subdir=None):
    out = {}
    for p in glob.glob(str(ROOT / pattern)):
        if "-profiler" in p:
            continue
        j = json.load(open(p, encoding="utf-8"))
        tm = j.get("test metrics", j.get("metrics", {}))
        out[_seed(p)] = {k: (float(v) if v is not None else None) for k, v in tm.items()}
    return out


def load_baseline_variant(ds, variant):
    """semba_ab 主基线（对照，非本模型）。"""
    out = {}
    for p in glob.glob(str(ROOT / f"results/semba_ab/raw_full/{variant}/{ds}_linksign_seed*.json")):
        out[_seed(p)] = json.load(open(p, encoding="utf-8")).get("metrics", {})
    return out


def stat(vals):
    a = np.array([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return None, None, 0
    return float(a.mean()), float(a.std(ddof=0)), int(a.size)


L = []
ap_ = L.append
csv_rows = []

ap_("T13 CNAS 行绝对值（linksign：w/o CNAS + CNAS-only；sign：w/o CNAS；mean±std(5 种子)）")
ap_("同批 full 对照 = E-2 raw_seeds / sign 同协议批；Δ = 行−full（‰）")
ap_("")

for ds in DS:
    ap_(f"===== {ds} =====")
    runs = {
        "full(同批)": load(f"results/E-2_ablation/raw_seeds/{ds}/*RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"),
        "w/o CNAS": load(f"results/E-2_ablation/raw_seeds/{ds}/*RAS-E.RASE-E.BTE-E.CNAS-D.P1.TE.json"),
        "CNAS-only": load(f"results/E-2_ablation/raw_seeds/{ds}/*RAS-D.RASE-D.BTE-D.CNAS-E.P1.TE.json"),
    }
    ap_(f"{'linksign':<12}{'row':<12}" + "".join(f"{k:>16}" for k in LINK_METRICS))
    ref = runs["full(同批)"]
    for name, per in runs.items():
        line = f"{'':<12}{name:<12}"
        for k in LINK_METRICS:
            vals = [m.get(k) for m in per.values()]
            mu, sd, n = stat(vals)
            line += f"{'—':>16}" if mu is None else f"{mu:.4f}±{sd:.4f}".rjust(16)
            csv_rows.append(("linksign", ds, name, k, mu, sd, n))
        ap_(line)
    # Δ（‰，仅均值差，配对需同 seed）
    for name in ("w/o CNAS", "CNAS-only"):
        per = runs[name]
        line = f"{'':<12}{'Δ'+name:<12}"
        for k in LINK_METRICS:
            d = [per[s][k] - ref[s][k] for s in per if s in ref and per[s].get(k) is not None and ref[s].get(k) is not None]
            line += f"{'—':>16}" if not d else f"{np.mean(d)*1000:>16.1f}"
        ap_(line)
    ap_("")

# sign 任务
ap_("===== sign（w/o CNAS）=====")
ap_(f"{'sign':<12}{'row':<12}" + "".join(f"{k:>11}" for k in SIGN_METRICS))
for ds in DS:
    per_full = load(f"results/sign_valthr/raw/{ds}/SignDyGFormer_seed*.json")
    per_woc = load(f"results/sign_wocnas/raw/{ds}/*CNAS-D.P1.TE.json")
    for name, per in (("full", per_full), ("w/o CNAS", per_woc)):
        line = f"{ds[:12]:<12}{name:<12}"
        for k in SIGN_METRICS:
            vals = [m.get(k) for m in per.values()]
            mu, sd, n = stat(vals)
            line += f"{'—':>11}" if mu is None else f"{mu:>8.4f}"
            csv_rows.append(("sign", ds, name, k, mu, sd, n))
        ap_(line)
    line = f"{'':<12}{'Δw/o CNAS':<12}"
    for k in SIGN_METRICS:
        d = [per_woc[s][k] - per_full[s][k] for s in per_woc if s in per_full and per_woc[s].get(k) is not None and per_full[s].get(k) is not None]
        line += f"{'—':>11}" if not d else f"{np.mean(d)*1000:>11.1f}"
    ap_(line)

out_txt = ROOT / "results/t13_cns_rows.txt"
out_txt.write_text("\n".join(L) + "\n", encoding="utf-8")
out_csv = ROOT / "results/t13_cns_rows.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["task", "dataset", "row", "metric", "mean", "std", "n"])
    for r in csv_rows:
        w.writerow([r[0], r[1], r[2], r[3], "" if r[4] is None else f"{r[4]:.6f}",
                    "" if r[5] is None else f"{r[5]:.6f}", r[6]])
print(f"写出 {out_txt}")
print(f"写出 {out_csv}")
print("\n".join(L))
