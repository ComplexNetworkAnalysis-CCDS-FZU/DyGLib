# -*- coding: utf-8 -*-
"""CNE-off 探针表：CNE（共同邻居编码器）开/关的同种子配对对比（linksign，5 种子）。

用法（仓库根）：
    python tools/verify/cne_off_table.py

数据：
- CNE-off：#136–145 产出，results/cne_off/raw/{ds}/SignDyGFormer_seed*.NN-Best.LF-Best.{mask}.P1.TE.CNE-D.json
- CNE-on 对照：results/E-2_ablation/raw_seeds/{ds}/SignDyGFormer_seed*.NN-Best.LF-Best.{mask}.P1.TE.json
（两条协议一致：同一 CLI train_sign_link_3class_prediction --ablation；仅
 --no-module-common-neighbor-encoder 开关不同；文件名仅差 .CNE-D 标记。）

语境（mask）：
- full = RAS-E.RASE-E.BTE-E.CNAS-E（其余模块全开，仅关 CNE）
- base = RAS-D.RASE-D.BTE-D.CNAS-D（全关 ⇒ base 相对：再加 CNE 通道）
Δ = CNE-on − CNE-off（= CNE 的贡献；Δ<0 表示关掉反而更好）。
"""
import glob
import json
import math
import pathlib
import re

from scipy import stats as _st

SEED_RE = re.compile(r"seed(\d+)")
DATASETS = ["RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha"]
CONTEXTS = [
    ("full", "RAS-E.RASE-E.BTE-E.CNAS-E"),
    ("base", "RAS-D.RASE-D.BTE-D.CNAS-D"),
]
KEYS = ["auc", "ap", "f1_wt", "f1_mac"]

ON_TPL = "results/E-2_ablation/raw_seeds/{ds}/SignDyGFormer_seed*.NN-Best.LF-Best.{mask}.P1.TE.json"
OFF_TPL = "results/cne_off/raw/{ds}/SignDyGFormer_seed*.NN-Best.LF-Best.{mask}.P1.TE.CNE-D.json"


def load(pattern: str, mask: str, suffix: str):
    out = {}
    for f in glob.glob(pattern):
        name = pathlib.Path(f).name
        if mask not in name or not name.endswith(suffix):
            continue
        sm = SEED_RE.search(name)
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


def paired(vals_a, vals_b):
    """同种子配对；返回 (Δmean, sd, t, df, p, d)。"""
    seeds = sorted(set(vals_a) & set(vals_b))
    if not seeds:
        return None
    ds = [vals_a[s] - vals_b[s] for s in seeds]
    n = len(ds)
    mu = sum(ds) / n
    if n == 1:
        return mu, None, None, 0, None, None
    sd = math.sqrt(sum((x - mu) ** 2 for x in ds) / (n - 1))
    tt = _st.ttest_rel([vals_a[s] for s in seeds], [vals_b[s] for s in seeds])
    d = mu / sd if sd > 0 else float("inf")
    return mu, sd, float(tt.statistic), n - 1, float(tt.pvalue), d


def main() -> None:
    summary = {}
    for ds in DATASETS:
        for ctx, mask in CONTEXTS:
            on = load(ON_TPL.format(ds=ds, mask=mask), mask, ".P1.TE.json")
            off = load(OFF_TPL.format(ds=ds, mask=mask), mask, ".P1.TE.CNE-D.json")
            seeds = sorted(set(on) & set(off))
            if not seeds:
                print(f"== {ds} · {ctx} == [空]（on={len(on)} off={len(off)}）\n")
                continue
            print(f"== {ds} · {ctx}（mask={mask}；n={len(seeds)}，seeds={seeds}）==")
            for k in KEYS:
                a = {s: float(on[s][k]) for s in seeds if k in on[s]}
                b = {s: float(off[s][k]) for s in seeds if k in off[s]}
                r = paired(a, b)
                if r is None:
                    continue
                mu, sd, t, df, p, d = r
                on_mu = sum(a.values()) / len(a)
                off_mu = sum(b.values()) / len(b)
                print(
                    f"  {k:<7} ON {on_mu:.4f}±{pstd(list(a.values())):.4f} | "
                    f"OFF {off_mu:.4f}±{pstd(list(b.values())):.4f} | "
                    f"Δ(on−off)={mu:+.4f} sd={sd:.4f} t={t:+.2f} df={df} "
                    f"p={p:.4f} d={d:+.2f}"
                )
            print()
            auc_r = paired(
                {s: float(on[s]["auc"]) for s in seeds},
                {s: float(off[s]["auc"]) for s in seeds},
            )
            summary[(ds, ctx)] = auc_r

    print("== 汇总（auc；Δ = CNE-on − CNE-off；负 = 去掉 CNE 更好）==")
    header = f"{'dataset':<22}" + f"{'full Δ (p)':<22}" + f"{'base Δ (p)':<22}"
    print(header)
    print("-" * len(header))
    for ds in DATASETS:
        cells = []
        for ctx, _ in CONTEXTS:
            r = summary.get((ds, ctx))
            if r is None:
                cells.append("-".ljust(20))
            else:
                mu, sd, t, df, p, d = r
                cells.append(f"{mu:+.4f} (p={p:.3f})".ljust(20))
        print(f"{ds:<22}" + "".join(cells))


if __name__ == "__main__":
    main()
