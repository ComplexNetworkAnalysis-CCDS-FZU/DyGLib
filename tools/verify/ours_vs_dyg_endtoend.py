# -*- coding: utf-8 -*-
"""ours vs 真 DyG：端到端指标全量对照（5 数据集 × 5 种子配对 t/p）。

口径（按 sign_f1 更正后的裁定：只看端到端键）：
  linksign: f1_wt / f1_mac / f1_mic / acc / auc / auc_wt / ap / mcc
  sign:     f1_macro / f1_binary / acc / auc / ap / mcc(若有)
  ours: raw_base（linksign）/ sign_valthr（sign）；DyG: s1_refresh。
输出：results/ours_vs_dyg_endtoend_20260925.txt
"""
from __future__ import annotations

import glob
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ["WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha", "BitcoinOTC"]
LINKSIGN_KEYS = ["f1_wt", "f1_mac", "f1_mic", "acc", "auc", "auc_wt", "ap", "mcc"]
SIGN_KEYS = ["f1_macro", "f1_binary", "acc", "auc", "ap"]


def _seed(p):
    return int(re.search(r"seed(\d+)", Path(p).name).group(1))


def load(pattern):
    out = {}
    for p in glob.glob(str(ROOT / pattern)):
        if "-profiler" in p:
            continue
        tm = json.load(open(p, encoding="utf-8"))["test metrics"]
        out[_seed(p)] = {k: float(v) for k, v in tm.items() if v is not None}
    return out


def paired(a, b, k):
    pairs = [(a[s][k], b[s][k]) for s in a if s in b and k in a[s] and k in b[s]]
    if not pairs:
        return None
    d = [x - y for x, y in pairs]
    m = statistics.mean(d) * 1000
    npos = sum(1 for x in d if x > 0)
    p = None
    try:
        from scipy import stats

        p = stats.ttest_rel([x for x, _ in pairs], [y for _, y in pairs]).pvalue
    except Exception:
        pass
    return m, npos, len(d), p


L = []
ap_ = L.append
ap_("ours vs DyGFormer 端到端指标对照（Δ‰ = ours − DyG；5 种子配对 t/p）")
ap_("ours：linksign=raw_base / sign=sign_valthr；DyG：s1_refresh（同数据同协议）")
ap_("")
for task, keys, opat, dpat in (
    ("linksign", LINKSIGN_KEYS,
     "results/e1a_tailfill/raw_base/linksign/{ds}/*.json",
     "results/s1_refresh/raw/linksign/{ds}/DyGFormer_seed*.json"),
    ("sign", SIGN_KEYS,
     "results/sign_valthr/raw/{ds}/SignDyGFormer_seed*.json",
     "results/s1_refresh/raw/sign/{ds}/DyGFormer_seed*.json"),
):
    ap_(f"===== {task} =====")
    for ds in DS:
        o = load(opat.format(ds=ds))
        d = load(dpat.format(ds=ds))
        cells = []
        for k in keys:
            r = paired(o, d, k)
            if r is None:
                cells.append(f"{k}: —")
                continue
            sig = "✅" if (r[3] is not None and r[3] < 0.05 and r[0] > 0) else ""
            cells.append(f"{k}: {r[0]:+.1f} ({r[1]}/{r[2]}){' p=%.4f' % r[3] if r[3] is not None else ''}{sig}")
        ap_(f"-- {ds}")
        for c in cells:
            ap_(f"     {c}")
    ap_("")

out = ROOT / "results/ours_vs_dyg_endtoend_20260925.txt"
out.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"写出 {out}")
print("\n".join(L))
