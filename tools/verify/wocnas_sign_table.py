# -*- coding: utf-8 -*-
"""sign 任务对照表：full（val-thr 刷新版）vs w/o CNAS（#150–154 新批）。

用法（仓库根）：
    python tools/verify/wocnas_sign_table.py

数据：
- full   = results/sign_valthr/raw/{ds}/SignDyGFormer_seed*.NN-{nn}.LF-{lf}.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json
- w/o CNAS = results/sign_wocnas/raw/{ds}/ 同名但 CNAS-D
约定：**Δ = full − w/o CNAS**（负 = 去掉 CNAS 更好）。
"""
import glob
import json
import math
import pathlib
import re

from scipy import stats as _st

SEED_RE = re.compile(r"seed(\d+)")
PARAMS = [
    ("RedditHyperlinkTitle", "100", "1"),
    ("RedditHyperlinkBody", "60", "1"),
    ("BitcoinAlpha", "40", "15"),
    ("BitcoinOTC", "40", "15"),
    ("WikiVote", "40", "15"),
]
KEYS = ["auc", "ap", "f1_binary", "f1_macro", "f1_weighted", "acc"]


def load(root: str, ds: str, nn: str, lf: str, cnas: str):
    pat = (
        root
        + "/"
        + ds
        + "/SignDyGFormer_seed*.NN-"
        + nn
        + ".LF-"
        + lf
        + ".RAS-E.RASE-E.BTE-E.CNAS-"
        + cnas
        + ".P1.TE.json"
    )
    out = {}
    for f in glob.glob(pat):
        sm = SEED_RE.search(pathlib.Path(f).name)
        if not sm:
            continue
        metrics = json.load(open(f, encoding="utf-8")).get("test metrics", {})
        out[int(sm.group(1))] = metrics
    return out


def pstd(vals):
    if not vals:
        return 0.0
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    ds = [a[s] - b[s] for s in seeds]
    n = len(ds)
    mu = sum(ds) / n
    if n == 1:
        return mu, None, None, 0, None, None
    sd = math.sqrt(sum((x - mu) ** 2 for x in ds) / (n - 1))
    tt = _st.ttest_rel([a[s] for s in seeds], [b[s] for s in seeds])
    d = mu / sd if sd > 0 else float("inf")
    return mu, sd, float(tt.statistic), n - 1, float(tt.pvalue), d


def main() -> None:
    summary = {}
    for ds, nn, lf in PARAMS:
        full = load("results/sign_valthr/raw", ds, nn, lf, "E")
        woc = load("results/sign_wocnas/raw", ds, nn, lf, "D")
        print(f"== {ds}（NN-{nn}/LF-{lf}；full n={len(full)}，w/o CNAS n={len(woc)}）==")
        for k in KEYS:
            a = {s: float(full[s][k]) for s in full if k in full[s]}
            b = {s: float(woc[s][k]) for s in woc if k in woc[s]}
            r = paired(a, b)
            if r is None:
                print(f"  {k:<12} [缺数据]")
                continue
            mu, sd, t, df, p, d = r
            a_mu = sum(a.values()) / len(a)
            b_mu = sum(b.values()) / len(b)
            print(
                f"  {k:<12} full {a_mu:.4f}±{pstd(list(a.values())):.4f} | "
                f"w/oCNAS {b_mu:.4f}±{pstd(list(b.values())):.4f} | "
                f"Δ(full−w/o)={mu:+.4f} sd={sd:.4f} t={t:+.2f} df={df} p={p:.4f} d={d:+.2f}"
            )
        summary[ds] = {
            k: paired(
                {s: float(full[s][k]) for s in full if k in full[s]},
                {s: float(woc[s][k]) for s in woc if k in woc[s]},
            )
            for k in KEYS
        }
        print()

    print("== 汇总：Δ = full − w/o CNAS（负 = 去掉更好）==")
    header = "dataset".ljust(22) + "".join(f"{k}".ljust(20) for k in ["auc", "f1_macro", "f1_weighted", "f1_binary"])
    print(header)
    print("-" * len(header))
    for ds, _, _ in PARAMS:
        cells = []
        for k in ["auc", "f1_macro", "f1_weighted", "f1_binary"]:
            r = summary[ds].get(k)
            if r is None:
                cells.append("-".ljust(18))
            else:
                mu, sd, t, df, p, d = r
                cells.append(f"{mu:+.4f} (p={p:.3f})".ljust(18))
        print(f"{ds:<22}" + "".join(cells))


if __name__ == "__main__":
    main()
