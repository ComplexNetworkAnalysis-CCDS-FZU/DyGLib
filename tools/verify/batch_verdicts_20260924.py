# -*- coding: utf-8 -*-
"""2026-09-24 四批判定总表：RB m=80 确认 / G1 / EVT 固定阈值 / sign 扩批。

输出：results/batch_verdicts_20260924.txt（人读）；用法：python tools/verify/batch_verdicts_20260924.py
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np

try:
    from scipy import stats as _st
except Exception:
    _st = None

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha", "BitcoinOTC"]
LINK_METRICS = ["auc", "f1_wt", "f1_mac", "ap"]
SIGN_METRICS = ["auc", "f1_macro", "f1_binary", "f1_weighted"]
OUT = []


def say(s=""):
    OUT.append(s)
    print(s)


def load(pattern: str) -> dict[int, dict]:
    out = {}
    for p in glob.glob(str(ROOT / pattern)):
        if "-profiler" in p:
            continue
        m = re.search(r"seed(\d+)", Path(p).name)
        out[int(m.group(1))] = json.load(open(p, encoding="utf-8"))["test metrics"]
    return out


def pair(base: dict, test: dict, metric: str):
    seeds = sorted(set(base) & set(test))
    if not seeds:
        return None
    b = np.array([float(base[s][metric]) for s in seeds])
    t = np.array([float(test[s][metric]) for s in seeds])
    d = t - b
    r = {"n": len(seeds), "base": b.mean(), "test": t.mean(), "d": d.mean(),
         "sd": d.std(ddof=1) if len(d) > 1 else float("nan"),
         "n_pos": int((d > 0).sum())}
    if len(d) > 1 and _st is not None:
        tt = _st.ttest_rel(t, b)
        r["t"], r["p"] = float(tt.statistic), float(tt.pvalue)
    else:
        r["t"] = r["p"] = None
    return r


def table(title, base_glob, test_glob, metrics, ds_list=DATASETS, tag="linksign"):
    say("=" * 110)
    say(title)
    say("=" * 110)
    for metric in metrics:
        say(f"\n--- {metric}（Δ = test − base；同种子配对）---")
        say(f"{'数据集':<22}{'base':>10}{'test':>10}{'Δ':>10}{'sd':>9}{'t':>7}{'p':>9}{'同向':>6}")
        for ds in ds_list:
            base = load(base_glob.format(ds=ds))
            test = load(test_glob.format(ds=ds))
            r = pair(base, test, metric)
            if r is None:
                say(f"{ds:<22}{'（缺）':>10}")
                continue
            txs = f"{r['t']:.2f}" if r["t"] is not None else "—"
            pxs = f"{r['p']:.4f}" if r["p"] is not None else "—"
            say(f"{ds:<22}{r['base']:>10.4f}{r['test']:>10.4f}{r['d']:>+10.4f}{r['sd']:>9.4f}{txs:>7}{pxs:>9}{str(r['n_pos'])+'/'+str(r['n']):>6}")
    say()


def main() -> int:
    # A. RB m=80 五种籽确认（test pattern 与 base 不在同一模板，单独处理）
    say("=" * 110)
    say("A. E1c RB m=80 五种籽确认（test=RK-80 ×5 vs base=full ×5）")
    say("=" * 110)
    for metric in LINK_METRICS:
        base = load("results/e1a_tailfill/raw_base/linksign/RedditHyperlinkBody/*.json")
        test = load("results/e1c_rb80/raw/*.json")
        r = pair(base, test, metric)
        txs = f"{r['t']:.2f}" if r["t"] is not None else "—"
        pxs = f"{r['p']:.4f}" if r["p"] is not None else "—"
        say(f"{metric:<10} base={r['base']:.4f} test={r['test']:.4f} Δ={r['d']:+.4f} sd={r['sd']:.4f} t={txs} p={pxs} 同向={r['n_pos']}/{r['n']}")
    say()

    # B. G1
    table(
        "B1. G1 linksign（G1 ×5 种子 vs full ×5 种子）",
        "results/e1a_tailfill/raw_base/linksign/{ds}/*.json",
        "results/g1_gate/raw/linksign/{ds}/*.json",
        LINK_METRICS,
    )
    table(
        "B2. G1 sign（G1 seed42 vs full seed42）",
        "results/e1a_tailfill/raw_base/sign/{ds}/*.json",
        "results/g1_gate/raw/sign/{ds}/*.json",
        SIGN_METRICS,
    )

    # C. EVT 固定阈值
    table(
        "C1. E1a @full-阈值（EVT；×5 种子；test=EVT vs base=full）——注意与 C2 对照看",
        "results/e1a_tailfill/raw_base/linksign/{ds}/*.json",
        "results/e1a_tailfill/evt/linksign/{ds}/*.json",
        LINK_METRICS,
    )
    table(
        "C2. E1a 自动阈值（raw；×5 种子；test=auto vs base=full）——原作判定用",
        "results/e1a_tailfill/raw_base/linksign/{ds}/*.json",
        "results/e1a_tailfill/raw/linksign/{ds}/*.json",
        LINK_METRICS,
    )
    # sign 侧（seed42 单点）
    say("=" * 110)
    say("C3. sign 侧固定阈值（seed42 单点；EVT vs auto vs full）")
    say("=" * 110)
    for ds in DATASETS:
        auto = load(f"results/e1a_tailfill/raw/sign/{ds}/*.json")
        evt = load(f"results/e1a_tailfill/evt/sign/{ds}/*.json")
        base = load(f"results/e1a_tailfill/raw_base/sign/{ds}/*.json")
        s = 42
        say(f"{ds:<22} auc: full={float(base[s]['auc']):.4f} auto={float(auto[s]['auc']):.4f} evt={float(evt[s]['auc']):.4f} | "
            f"f1_macro: full={float(base[s]['f1_macro']):.4f} auto={float(auto[s]['f1_macro']):.4f} evt={float(evt[s]['f1_macro']):.4f}")
    say()

    # D. sign 扩批（5×5）
    table(
        "D. sign 扩批（B：`.TF-E` ×5 种子 vs full ×5 种子；sign_valthr）",
        "results/sign_valthr/raw/{ds}/*.json",
        "results/sign_tailfill5/raw/{ds}/*.json",
        SIGN_METRICS,
    )

    (ROOT / "results/batch_verdicts_20260924.txt").write_text("\n".join(OUT) + "\n", encoding="utf-8")
    print("[ok] results/batch_verdicts_20260924.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
