"""valthr_sign_report.py — val-thr 刷新批交付报表（2026-09-17）。

对象：
- **ours sign**（#129–133；full 口径 RAS-E.RASE-E.BTE-E.CNAS-E）：
  ① 新旧同名文件对照（auc/ap/f1_binary/f1_weighted/acc；sha256 同/异计数）；
- **DyG 真基线**（#121–128）：② sign（RT/RB 为 val-thr 刷新版）与 linksign 五数据集汇总；
  RT/RB sign 新旧对照；
- ③ ours(新) − DyG(新) sign 同种子配对统计（每数据集 5 指标）；
- ④ ours linksign full − DyG linksign 同种子配对（auc / f1_wt）。

指标键：sign = auc/ap/f1_binary/f1_macro/f1_weighted/acc；linksign = …/f1_wt/f1_mac/auc/ap。
口径注：val-thr 下 **auc/ap 与旧（thr=0.5）逐位相同**（训练确定，仅阈值相关指标变化）。

用法（仓库根）：python tools/verify/valthr_sign_report.py
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATASETS = [
    "RedditHyperlinkTitle",
    "RedditHyperlinkBody",
    "BitcoinAlpha",
    "BitcoinOTC",
    "WikiVote",
]
SIGN_KEYS = ["auc", "ap", "f1_binary", "f1_weighted", "acc"]
LINKSIGN_KEYS = ["auc", "f1_wt", "ap", "f1_mac"]
SEED_RE = re.compile(r"seed(\d+)")

OURS_OLD = ROOT / "results/main_tables/raw/sign/{ds}"
OURS_NEW = ROOT / "results/sign_valthr/raw/{ds}"
DYG_SIGN = ROOT / "results/s1_refresh/raw/sign/{ds}"
DYG_SIGN_NEW = ROOT / "results/s1_refresh/raw_valthr/sign/{ds}"
DYG_LINKSIGN = ROOT / "results/s1_refresh/raw/linksign/{ds}"
OURS_LINKSIGN = ROOT / "results/E-2_ablation/raw_seeds/{ds}"
E2S_FULL_PAT = "SignDyGFormer_seed*.NN-Best.LF-Best.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"

try:
    from scipy import stats as _st
except Exception:  # pragma: no cover
    _st = None


def pstd(vals):
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    ds = [a[s] - b[s] for s in seeds]
    n = len(ds)
    mu = sum(ds) / n
    if n < 2:
        return mu, None, None
    sd = math.sqrt(sum((x - mu) ** 2 for x in ds) / (n - 1))
    if _st is not None:
        tt = _st.ttest_rel([a[s] for s in seeds], [b[s] for s in seeds])
        t, p = float(tt.statistic), float(tt.pvalue)
    else:
        t = mu / (sd / math.sqrt(n)) if sd > 0 else float("inf")
        p = None
    return mu, t, p


def load_seed_dir(d: pathlib.Path, pat: str = "*.P1.TE.json"):
    """{seed: (file, metrics)}。"""
    out = {}
    if not d.is_dir():
        return out
    for f in sorted(d.glob(pat)):
        m = SEED_RE.search(f.name)
        if not m:
            continue
        tm = json.loads(f.read_text(encoding="utf-8")).get("test metrics", {})
        out[int(m.group(1))] = (f, tm)
    return out


def series(seedmap, key):
    return {s: float(tm[key]) for s, (_f, tm) in seedmap.items() if key in tm}


def ms(vals):
    return f"{sum(vals)/len(vals):.4f}±{pstd(vals):.4f}"


def fmt_pair(old, new):
    return f"{old:.4f}→{new:.4f}"


def main():
    print("=" * 100)
    print("① ours sign 新旧对照（#129–133；full 口径；同种子 n=5）")
    print("=" * 100)
    hdr = f"{'数据集':<20s} {'auc old→new':<22s} {'f1_bin old→new':<24s} {'f1_wt old→new':<24s} {'acc old→new':<24s} sha同/异"
    print(hdr)
    print("-" * 118)
    for ds in DATASETS:
        old_dir = pathlib.Path(str(OURS_OLD).format(ds=ds))
        new = load_seed_dir(pathlib.Path(str(OURS_NEW).format(ds=ds)))
        if not new:
            print(f"{ds:<20s} [缺新档]")
            continue
        # 只对照同名文件（旧目录可能含多套参数 → 按全量文件名建索引，勿用 seed 索引）
        old_byname = {}
        if old_dir.is_dir():
            for f in sorted(old_dir.glob("*.P1.TE.json")):
                m = SEED_RE.search(f.name)
                if not m:
                    continue
                tm = json.loads(f.read_text(encoding="utf-8")).get("test metrics", {})
                old_byname[f.name] = (int(m.group(1)), f, tm)
        sha_same = sha_diff = 0
        old_sel, new_sel = {}, {}
        for s, (f, tm) in new.items():
            if f.name in old_byname:
                os_, of, otm = old_byname[f.name]
                same = hashlib.sha256(of.read_bytes()).hexdigest() == hashlib.sha256(f.read_bytes()).hexdigest()
                sha_same += int(same)
                sha_diff += int(not same)
                old_sel[os_], new_sel[s] = (of, otm), (f, tm)
        def cell(key):
            o = series(old_sel, key)
            n = series(new_sel, key)
            if not o or not n:
                return "-"
            return f"{fmt_pair(sum(o.values())/len(o), sum(n.values())/len(n))} (Δ{sum(n.values())/len(n)-sum(o.values())/len(o):+.4f})"
        print(f"{ds:<20s} {cell('auc'):<22s} {cell('f1_binary'):<24s} {cell('f1_weighted'):<24s} {cell('acc'):<24s} {sha_same}/{sha_diff}")
    print()

    print("=" * 100)
    print("② DyG 真基线（#121–128；5 种子 mean±pstd；RT/RB sign 为 val-thr 刷新版）")
    print("=" * 100)
    for task, keys, sub in (("sign", SIGN_KEYS, "sign"), ("linksign", LINKSIGN_KEYS, "linksign")):
        print(f"\n[{task}] {'数据集':<20s} " + "  ".join(f"{k:>17s}" for k in keys))
        for ds in DATASETS:
            if task == "sign" and ds in ("RedditHyperlinkTitle", "RedditHyperlinkBody"):
                d = pathlib.Path(str(DYG_SIGN_NEW).format(ds=ds))
            else:
                d = pathlib.Path(str(DYG_LINKSIGN if task == "linksign" else DYG_SIGN).format(ds=ds))
            sm = load_seed_dir(d)
            if not sm:
                print(f"      {ds:<20s} [缺档] {d}")
                continue
            cells = [ms(list(series(sm, k).values())) if series(sm, k) else "-" for k in keys]
            print(f"      {ds:<20s} " + "  ".join(f"{c:>17s}" for c in cells))
    print()
    print("  DyG sign RT/RB 新旧（val-thr 刷新）对照：")
    for ds in ("RedditHyperlinkTitle", "RedditHyperlinkBody"):
        o = load_seed_dir(pathlib.Path(str(DYG_SIGN).format(ds=ds)))
        n = load_seed_dir(pathlib.Path(str(DYG_SIGN_NEW).format(ds=ds)))
        if not o or not n:
            print(f"    {ds}: [缺档]")
            continue
        same = sum(
            1
            for s in set(o) & set(n)
            if hashlib.sha256(o[s][0].read_bytes()).hexdigest() == hashlib.sha256(n[s][0].read_bytes()).hexdigest()
        )
        for k in ("auc", "f1_binary", "f1_weighted", "acc"):
            ov, nv = series(o, k), series(n, k)
            if ov and nv:
                print(f"    {ds:<22s} {k:<12s} {fmt_pair(sum(ov.values())/len(ov), sum(nv.values())/len(nv))}"
                      f" (Δ{sum(nv.values())/len(nv)-sum(ov.values())/len(ov):+.4f})")
        print(f"    {ds:<22s} sha 相同 {same}/{len(set(o)&set(n))}")
    print()

    print("=" * 100)
    print("③ ours(新) − DyG(新) sign 同种子配对（n=5）")
    print("=" * 100)
    for ds in DATASETS:
        ours = load_seed_dir(pathlib.Path(str(OURS_NEW).format(ds=ds)))
        d = pathlib.Path(str(DYG_SIGN_NEW if ds in ("RedditHyperlinkTitle", "RedditHyperlinkBody") else DYG_SIGN).format(ds=ds))
        dyg = load_seed_dir(d)
        if not ours or not dyg:
            print(f"{ds}: [缺档]")
            continue
        print(f"\n[{ds}]")
        for k in SIGN_KEYS:
            r = paired(series(ours, k), series(dyg, k))
            if r is None:
                continue
            mu, t, p = r
            if t is None:
                print(f"    {k:<12s} Δ={mu:+.4f}")
            else:
                pstr = f"p={p:.4f}" if p is not None else "p=?"
                print(f"    {k:<12s} Δ={mu:+.4f} t={t:+.2f} {pstr}")
    print()

    print("=" * 100)
    print("④ ours linksign full − DyG linksign 同种子配对（n=5；auc / f1_wt）")
    print("=" * 100)
    for ds in DATASETS:
        ours = load_seed_dir(pathlib.Path(str(OURS_LINKSIGN).format(ds=ds)), E2S_FULL_PAT)
        dyg = load_seed_dir(pathlib.Path(str(DYG_LINKSIGN).format(ds=ds)))
        if not ours or not dyg:
            print(f"{ds}: [缺档 ours={bool(ours)} dyg={bool(dyg)}]")
            continue
        line = f"[{ds}]"
        for k in ("auc", "f1_wt"):
            r = paired(series(ours, k), series(dyg, k))
            if r is None:
                continue
            mu, t, p = r
            pstr = f"p={p:.4f}" if p is not None else "p=?"
            line += f"  {k}: Δ={mu:+.4f} t={t:+.2f} {pstr}"
        print(line)


if __name__ == "__main__":
    sys.exit(main())
