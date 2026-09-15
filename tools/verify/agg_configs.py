"""agg_configs.py — 按配置旗标聚合结果 JSON 的指标（mean±pstd），可选同种子配对差值。

用法（仓库根目录）：
    python tools/verify/agg_configs.py results/E-2_ablation/raw_seeds/BitcoinAlpha
    python tools/verify/agg_configs.py "results/E-2_ablation/raw_seeds/*"
    python tools/verify/agg_configs.py "results/E-2_ablation/raw_seeds/*" \
        --pair RAS-D.RASE-D.BTE-E.CNAS-D RAS-E.RASE-E.BTE-E.CNAS-E   # BTE-only − full（同种子）

说明：
- 文件名解析：seed{N}；配置旗标 RAS-[ED].RASE-[ED].BTE-[ED].CNAS-[ED]；指标取 JSON 的 "test metrics"。
- std 口径：pstd（ddof=0），与全仓库一致；配对 t 用样本 sd（df=n−1）。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import pathlib
import re

DEFAULT_KEYS = ["auc", "ap", "sign_f1", "f1_mac"]
FLAG_RE = re.compile(r"RAS-([ED])\.RASE-([ED])\.BTE-([ED])\.CNAS-([ED])")
SEED_RE = re.compile(r"seed(\d+)")

CONFIG_ORDER = [
    "RAS-E.RASE-E.BTE-E.CNAS-E",  # full
    "RAS-E.RASE-D.BTE-E.CNAS-E",  # +RAS
    "RAS-D.RASE-E.BTE-E.CNAS-E",  # +RAE
    "RAS-D.RASE-D.BTE-E.CNAS-E",  # base
    "RAS-D.RASE-D.BTE-D.CNAS-E",  # CNAS-only
    "RAS-D.RASE-D.BTE-E.CNAS-D",  # BTE-only
    "RAS-D.RASE-D.BTE-D.CNAS-D",  # vanilla
]


def parse_dir(d: pathlib.Path, keys):
    """返回 {cfg: {key: [(seed, value)]}}。"""
    out = {}
    for f in sorted(d.glob("*.P1.TE.json")):
        m = FLAG_RE.search(f.name)
        if not m:
            continue
        cfg = m.group(0)
        sm = SEED_RE.search(f.name)
        seed = int(sm.group(1)) if sm else -1
        try:
            tm = json.loads(f.read_text(encoding="utf-8")).get("test metrics", {})
        except Exception:
            continue
        for k in keys:
            if k in tm:
                out.setdefault(cfg, {}).setdefault(k, []).append((seed, float(tm[k])))
    return out


def pstd(vals):
    if not vals:
        return 0.0
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="目录或带 * 的 glob（每个目录=一个数据集）")
    ap.add_argument("--keys", nargs="*", default=DEFAULT_KEYS)
    ap.add_argument(
        "--pair",
        nargs=2,
        metavar=("CFG_A", "CFG_B"),
        help="打印配对差值 A−B（同种子）按数据集汇总",
    )
    args = ap.parse_args()

    dirs = []
    for p in args.paths:
        hits = glob.glob(p) if any(ch in p for ch in "*?[") else [p]
        dirs += [pathlib.Path(h) for h in hits]
    dirs = [d for d in dirs if d.is_dir()]

    for d in dirs:
        data = parse_dir(d, args.keys)
        if not data:
            continue
        print(f"\n== {d.name} ==")
        print(f"{'config':<28s} {'n':>2s}  " + "  ".join(f"{k:>17s}" for k in args.keys))
        print("-" * (34 + 19 * len(args.keys)))
        for cfg in sorted(
            data, key=lambda c: (CONFIG_ORDER.index(c) if c in CONFIG_ORDER else 99, c)
        ):
            per = data[cfg]
            n = max(len(v) for v in per.values())
            cells = []
            for k in args.keys:
                vals = [v for _, v in per.get(k, [])]
                cells.append(
                    f"{sum(vals)/len(vals):.4f}±{pstd(vals):.4f}" if vals else "-"
                )
            print(f"{cfg:<28s} {n:>2d}  " + "  ".join(f"{c:>17s}" for c in cells))
        if args.pair:
            a, b = args.pair
            if a in data and b in data:
                print(f"-- 配对差值 {a} − {b}（同种子）--")
                for k in args.keys:
                    da = dict(data[a].get(k, []))
                    db = dict(data[b].get(k, []))
                    seeds = sorted(set(da) & set(db))
                    if not seeds:
                        continue
                    deltas = [da[s] - db[s] for s in seeds]
                    mu = sum(deltas) / len(deltas)
                    sd = (
                        math.sqrt(sum((x - mu) ** 2 for x in deltas) / (len(deltas) - 1))
                        if len(deltas) > 1
                        else 0.0
                    )
                    t = mu / (sd / math.sqrt(len(deltas))) if sd > 0 else float("inf")
                    print(
                        f"  {k:<10s} Δ={mu:+.4f} (sd={sd:.4f}, n={len(deltas)}, t={t:+.2f})"
                    )


if __name__ == "__main__":
    main()
