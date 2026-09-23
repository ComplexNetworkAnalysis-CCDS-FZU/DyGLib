# -*- coding: utf-8 -*-
"""E1a（CNAS 空白填补 `.TF-E`）正式批配对判定表。

读 `results/e1a_tailfill/raw/linksign/{ds}`（5×5）与 `raw/sign/{ds}`（5×1），
对照 `raw_base/...` 同名 full run（同配置、稳定代）：
- linksign（3class）：auc / f1_wt（主）/ f1_mac / ap，5 种子配对 t/p；
- sign（binary）：auc / f1_macro（主）/ f1_binary / f1_weighted，seed42 单点 Δ。

判据（Paper）：Δ≥0.005 + 配对显著 + ≥3/5 同向（sign 单种子仅报 Δ）。
用法（仓库根）：python tools/verify/e1a_pair_table.py
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np

try:
    from scipy import stats as _st
except Exception:  # pragma: no cover
    _st = None

ROOT = Path(__file__).resolve().parents[2]
LINKSIGN_METRICS = ["auc", "f1_wt", "f1_mac", "ap"]
SIGN_METRICS = ["auc", "f1_macro", "f1_binary", "f1_weighted"]
DATASETS = ["WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha", "BitcoinOTC"]


def load_run(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["test metrics"]


def load_side(sub: str, task: str, ds: str) -> dict:
    """返回 {seed: {metric: value}}。"""
    out = {}
    for p in glob.glob(str(ROOT / f"results/e1a_tailfill/{sub}/{task}/{ds}/*.json")):
        name = Path(p).name
        if "-profiler" in name:
            continue
        m = re.search(r"seed(\d+)", name)
        seed = int(m.group(1)) if m else -1
        out[seed] = load_run(Path(p))
    return out


def fmt(x: float) -> str:
    return f"{x:.4f}"


def paired(base: dict, test: dict, metric: str):
    seeds = sorted(set(base) & set(test))
    if not seeds:
        return None
    b = np.array([float(base[s][metric]) for s in seeds])
    t = np.array([float(test[s][metric]) for s in seeds])
    d = t - b
    res = {
        "n": len(seeds), "seeds": seeds, "base": b, "e1a": t, "d": d,
        "base_m": b.mean(), "e1a_m": t.mean(), "d_m": d.mean(),
        "d_sd": d.std(ddof=1) if len(d) > 1 else float("nan"),
    }
    if len(d) > 1 and _st is not None:
        tt = _st.ttest_rel(t, b)
        res["t"] = float(tt.statistic)
        res["p"] = float(tt.pvalue)
    else:
        res["t"] = res["p"] = None
    res["n_pos"] = int((d > 0).sum())
    res["n_neg"] = int((d < 0).sum())
    return res


def main() -> int:
    print("=" * 100)
    print("E1a（.TF-E 空白填补）正式批配对判定 —— linksign（3class，5 数据集 × 5 种子）")
    print("=" * 100)
    verdict_rows = []
    for metric in LINKSIGN_METRICS:
        print(f"\n--- 指标：{metric}（Δ = E1a − full）---")
        print(f"{'数据集':<22}{'full(mean±pstd)':<22}{'E1a(mean±pstd)':<22}{'Δmean':>9}{'sd':>8}{'t':>7}{'p':>9}{'同向':>7}")
        ds_means = []
        for ds in DATASETS:
            base = load_side("raw_base", "linksign", ds)
            test = load_side("raw", "linksign", ds)
            r = paired(base, test, metric)
            if r is None:
                print(f"{ds:<22}{'（缺文件）':<22}")
                continue
            ds_means.append(r["d_m"])
            txs = f"{r['t']:.2f}" if r["t"] is not None else "—"
            pxs = f"{r['p']:.4f}" if r["p"] is not None else "—"
            bstr = f"{r['base_m']:.4f}±{r['base'].std(ddof=0):.4f}"
            estr = f"{r['e1a_m']:.4f}±{r['e1a'].std(ddof=0):.4f}"
            dirstr = f"{r['n_pos']}/{r['n']}"
            print(f"{ds:<22}{bstr:<22}{estr:<22}{r['d_m']:>+9.4f}{r['d_sd']:>8.4f}{txs:>7}{pxs:>9}{dirstr:>7}")
            verdict_rows.append((metric, ds, r))
        if ds_means:
            print(f"{'跨数据集平均':<22}{'':<22}{'':<22}{np.mean(ds_means):>+9.4f}")

    print("\n" + "=" * 100)
    print("E1a（.TF-E）正式批 —— sign（binary，5 数据集 × seed42；单点 Δ）")
    print("=" * 100)
    for metric in SIGN_METRICS:
        print(f"\n--- 指标：{metric}（Δ = E1a − full，seed42）---")
        print(f"{'数据集':<22}{'full':>10}{'E1a':>10}{'Δ':>10}")
        for ds in DATASETS:
            base = load_side("raw_base", "sign", ds)
            test = load_side("raw", "sign", ds)
            seeds = sorted(set(base) & set(test))
            if not seeds:
                print(f"{ds:<22}{'（缺文件）':>10}")
                continue
            s = seeds[0]
            d = float(test[s][metric]) - float(base[s][metric])
            print(f"{ds:<22}{float(base[s][metric]):>10.4f}{float(test[s][metric]):>10.4f}{d:>+10.4f}")

    print("\n" + "=" * 100)
    print("判据核对（linksign：Δ≥+0.005 且 p<0.05 且 ≥3/5 同向；主指标 f1_wt / 次 auc）")
    print("=" * 100)
    for metric in ["f1_wt", "auc", "f1_mac"]:
        hits = []
        for (m, ds, r) in verdict_rows:
            if m != metric:
                continue
            ok = r["d_m"] >= 0.005 and (r["p"] is not None and r["p"] < 0.05) and r["n_pos"] >= 3
            flag = "✅" if ok else ("△" if r["d_m"] >= 0.005 else "✗")
            hits.append(f"{ds} Δ{r['d_m']:+.4f}{flag}")
        print(f"{metric:<8}: " + "  ".join(hits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
