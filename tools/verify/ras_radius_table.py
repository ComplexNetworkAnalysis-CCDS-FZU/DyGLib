# -*- coding: utf-8 -*-
"""双半径验证批表格：扫描带 `.RLF-{k_r}` 标记的结果 JSON，按数据集输出 k_r 曲线。

用法（仓库根）：
    python tools/verify/ras_radius_table.py                          # 默认 results/ras_radius/raw/*/*
    python tools/verify/ras_radius_table.py "results/ras_radius/raw5/*"
    python tools/verify/ras_radius_table.py --keys auc ap f1_wt f1_mac <globs...>

输出：每个数据集一张表（k_r 升序；多种子自动聚合 mean±pstd(ddof=0)，n=种子数）；
`*=k_c` 标该数据集对角点（k_r == k_c，即原公式）；`best` 标 auc 均值最大行；
表尾给出「best vs 对角」同种子配对统计（Δ/sd/t/df/p/Cohen's d；单种子时退化为点差）。
文件名约定：`...seed{S}.NN-{n}.LF-{k_c}.RLF-{k_r}.RAS-...`（k_r 显式给出时才会带 .RLF-）。
"""
import argparse
import glob
import json
import math
import pathlib
import re

try:  # 可选依赖：有 scipy 时用 ttest_rel（与仓库统计口径一致）
    from scipy import stats as _st
except Exception:  # pragma: no cover
    _st = None

DEFAULT_KEYS = ["auc", "ap", "f1_wt", "f1_mac"]
FNAME_RE = re.compile(r"NN-(\d+)\.LF-(\d+)\.RLF-(\d+)")
SEED_RE = re.compile(r"seed(\d+)")


def pstd(vals):
    if not vals:
        return 0.0
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))


def paired(vals_a, vals_b):
    """同种子配对统计；返回 (Δmean, sd, t, df, p, d)；单种子时仅给点差。"""
    seeds = sorted(set(vals_a) & set(vals_b))
    if not seeds:
        return None
    ds = [vals_a[s] - vals_b[s] for s in seeds]
    n = len(ds)
    mu = sum(ds) / n
    if n == 1:
        return mu, None, None, 0, None, None
    sd = math.sqrt(sum((x - mu) ** 2 for x in ds) / (n - 1))
    if _st is not None:
        tt = _st.ttest_rel([vals_a[s] for s in seeds], [vals_b[s] for s in seeds])
        t, p = float(tt.statistic), float(tt.pvalue)
    else:
        t = mu / (sd / math.sqrt(n)) if sd > 0 else float("inf")
        p = None
    d = mu / sd if sd > 0 else float("inf")
    return mu, sd, t, n - 1, p, d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="*", default=None)
    ap.add_argument("--keys", nargs="+", default=DEFAULT_KEYS)
    ap.add_argument(
        "--pair",
        nargs=2,
        type=int,
        default=None,
        metavar=("KR_A", "KR_B"),
        help="额外打印指定两档 k_r 的同种子配对统计（每个数据集）",
    )
    args = ap.parse_args()
    globs = args.globs or ["results/ras_radius/raw/*/*"]

    files = []
    for g in globs:
        files += glob.glob(g)

    # {(ds, k_r): {"kc": k_c, "nn": NN, "seeds": {seed: metrics}}}
    by_ds = {}
    for f in sorted(files):
        name = pathlib.Path(f).name
        m = FNAME_RE.search(name)
        if not m:
            continue
        n, kc, kr = (int(x) for x in m.groups())
        sm = SEED_RE.search(name)
        seed = int(sm.group(1)) if sm else -1
        metrics = json.load(open(f, encoding="utf-8")).get("test metrics", {})
        ds = pathlib.Path(f).parent.name
        rec = by_ds.setdefault((ds, kr), {"kc": kc, "nn": n, "seeds": {}})
        rec["seeds"][seed] = metrics

    if not by_ds:
        print("[空] 没有匹配的 .RLF- 结果文件")
        return

    for ds in sorted({d for d, _ in by_ds}):
        rows = sorted(kr for d, kr in by_ds if d == ds)
        print(f"== {ds} ==")
        w = max(len(k) for k in args.keys)
        header = (
            "k_r".ljust(5)
            + " n |"
            + " | ".join(k.ljust(w + 10) for k in args.keys)
            + " | note"
        )
        print(header)
        print("-" * len(header))

        # 先算 auc 均值定 best 行、找对角点
        auc_means = {}
        diag_k = None
        for kr in rows:
            rec = by_ds[(ds, kr)]
            if kr == rec["kc"]:
                diag_k = kr
            vals = [float(v["auc"]) for v in rec["seeds"].values() if "auc" in v]
            if vals:
                auc_means[kr] = sum(vals) / len(vals)
        best_k = max(auc_means, key=auc_means.get) if auc_means else None

        for kr in rows:
            rec = by_ds[(ds, kr)]
            nseed = len(rec["seeds"])
            cells = []
            for k in args.keys:
                vals = [float(v[k]) for v in rec["seeds"].values() if k in v]
                if not vals:
                    cells.append("-".ljust(w + 10))
                elif nseed == 1:
                    cells.append(f"{vals[0]:.4f}".ljust(w + 10))
                else:
                    mu = sum(vals) / len(vals)
                    cells.append(f"{mu:.4f}±{pstd(vals):.4f}".ljust(w + 10))
            note = []
            if kr == rec["kc"]:
                note.append("*=k_c")
            if kr == best_k:
                note.append("best")
            print(f"{kr:<5} {nseed} | " + " | ".join(cells) + " | " + " ".join(note))

        def report_pair(ka, kb, label):
            if (ds, ka) not in by_ds or (ds, kb) not in by_ds:
                return
            print(f"  {label}：k_r={ka} − k_r={kb}（同种子配对）")
            for k in args.keys:
                a = {
                    s: float(v[k])
                    for s, v in by_ds[(ds, ka)]["seeds"].items()
                    if k in v
                }
                b = {
                    s: float(v[k])
                    for s, v in by_ds[(ds, kb)]["seeds"].items()
                    if k in v
                }
                r = paired(a, b)
                if r is None:
                    continue
                mu, sd, t, df, p, d = r
                if t is None:
                    print(f"    {k:<8} Δ={mu:+.4f}（单种子点差）")
                else:
                    pstr = f"p={p:.4f}" if p is not None else "p=?"
                    print(
                        f"    {k:<8} Δ={mu:+.4f} sd={sd:.4f} t={t:+.2f} "
                        f"df={df} {pstr} d={d:+.2f}"
                    )

        if best_k is not None and diag_k is not None and best_k != diag_k:
            report_pair(best_k, diag_k, f"best(k_r={best_k}) vs 对角(k_r={diag_k})")
        if args.pair:
            report_pair(args.pair[0], args.pair[1], "指定配对")
        print()


if __name__ == "__main__":
    main()
