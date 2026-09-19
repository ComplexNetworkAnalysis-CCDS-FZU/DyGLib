# -*- coding: utf-8 -*-
"""LOO 满表：full vs 4 个 leave-one-out 掩码（linksign，5 种子同种子配对）。

用法（仓库根）：
    python tools/verify/loo_full_table.py

数据（本地归档 results/E-2_ablation/raw_seeds/{ds}）：
- full = RAS-E.RASE-E.BTE-E.CNAS-E（e2s 批 5 种子）
- idx7 = RAS-E.RASE-E.BTE-E.CNAS-D（w/o CNAS；本批 #146 重跑）
- idx8 = RAS-E.RASE-E.BTE-D.CNAS-E（w/o BTE；本批 #146 重跑）
- idx1 = RAS-E.RASE-D.BTE-E.CNAS-E（w/o RAE；#147 本批 + RT 用 e2c 批）
- idx2 = RAS-D.RASE-E.BTE-E.CNAS-E（w/o RAS；#147 本批 + RT 用 e2c 批）

约定：**Δ = full − mask**（正 = 该模块有正贡献；负 = 去掉更好）。
"""
import glob
import json
import math
import pathlib
import re

from scipy import stats as _st

SEED_RE = re.compile(r"seed(\d+)")
DATASETS = ["RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha", "BitcoinOTC", "WikiVote"]
FULL = "RAS-E.RASE-E.BTE-E.CNAS-E"
MASKS = [
    ("w/o CNAS", "RAS-E.RASE-E.BTE-E.CNAS-D"),
    ("w/o BTE", "RAS-E.RASE-E.BTE-D.CNAS-E"),
    ("w/o RAE", "RAS-E.RASE-D.BTE-E.CNAS-E"),
    ("w/o RAS", "RAS-D.RASE-E.BTE-E.CNAS-E"),
]
KEYS = ["auc", "f1_wt", "f1_mac"]


def load(ds: str, mask: str):
    pat = (
        "results/E-2_ablation/raw_seeds/"
        + ds
        + "/SignDyGFormer_seed*.NN-Best.LF-Best."
        + mask
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
    for ds in DATASETS:
        full = load(ds, FULL)
        print(f"== {ds} ==")
        w = 20
        header = "config".ljust(10) + "n |" + " | ".join(k.ljust(w) for k in KEYS)
        print(header)
        print("-" * len(header))
        for label, mask in [("full", FULL)] + MASKS:
            rec = full if label == "full" else load(ds, mask)
            n = len(rec)
            cells = []
            for k in KEYS:
                vals = [float(v[k]) for v in rec.values() if k in v]
                if not vals:
                    cells.append("-".ljust(w))
                else:
                    mu = sum(vals) / len(vals)
                    cells.append(f"{mu:.4f}±{pstd(vals):.4f}".ljust(w))
            print(f"{label:<10} {n} |" + " | ".join(cells))
        for label, mask in MASKS:
            rec = load(ds, mask)
            line = f"  Δ(full−{label})："
            parts = []
            for k in KEYS:
                a = {s: float(full[s][k]) for s in full if k in full[s]}
                b = {s: float(rec[s][k]) for s in rec if k in rec[s]}
                r = paired(a, b)
                if r is None:
                    parts.append(f"{k}=NA")
                    continue
                mu, sd, t, df, p, d = r
                if t is None:
                    parts.append(f"{k}: Δ={mu:+.4f}（单种子）")
                else:
                    parts.append(f"{k}: Δ={mu:+.4f} t={t:+.2f} p={p:.3f} d={d:+.2f}")
            print(line + " | ".join(parts))
            summary[(ds, label)] = paired(
                {s: float(full[s]["auc"]) for s in full},
                {s: float(rec[s]["auc"]) for s in rec},
            )
        print()

    print("== 汇总：Δauc = full − mask（负 = 去掉更好；p 为同种子配对）==")
    header = "dataset".ljust(22) + "".join(f"{label}".ljust(24) for label, _ in MASKS)
    print(header)
    print("-" * len(header))
    for ds in DATASETS:
        cells = []
        for label, _ in MASKS:
            r = summary.get((ds, label))
            if r is None:
                cells.append("-".ljust(22))
            else:
                mu, sd, t, df, p, d = r
                cells.append(f"{mu:+.4f} (p={p:.3f})".ljust(22))
        print(f"{ds:<22}" + "".join(cells))
    mean_cells = []
    for label, _ in MASKS:
        rs = [summary[(ds, label)] for ds in DATASETS if (ds, label) in summary]
        if rs:
            mu = sum(r[0] for r in rs) / len(rs)
            mean_cells.append(f"{mu:+.4f}".ljust(22))
        else:
            mean_cells.append("-".ljust(22))
    print("-" * len(header))
    print(f"{'跨数据集均值':<22}" + "".join(mean_cells))


if __name__ == "__main__":
    main()
