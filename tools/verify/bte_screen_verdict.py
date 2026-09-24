# -*- coding: utf-8 -*-
"""BTE 微调屏幕批判定（2026-09-25）：B2（linksign+sign）与 sign×w/o BTE。

对比（seed42 单点；测试批 vs 同代 full 基线）：
  A. B2 linksign：results/bte_stage1/raw/linksign vs results/e1a_tailfill/raw_base/linksign
  B. B2 sign：results/bte_stage1/raw/sign vs results/sign_valthr/raw
  C. sign w/o BTE：results/sign_nobte/raw vs results/sign_valthr/raw
输出：results/bte_screen_verdict_20260925.txt
用法：python tools/verify/bte_screen_verdict.py
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ["WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha", "BitcoinOTC"]
LINK = ["f1_wt", "f1_mac", "sign_f1", "ap", "auc"]
SIGN = ["auc", "f1_macro", "f1_binary"]


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


L = []
ap_ = L.append
ap_("BTE 屏幕批判定（seed42；Δ‰ 正 = 测试批更高；2026-09-25）")
ap_("")

ap_("== A. B2（linksign seed42） vs full ==")
ap_(f"{'ds':<22}" + "".join(f"{k:>12}" for k in LINK))
for ds in DS:
    test = load(f"results/bte_stage1/raw/linksign/{ds}/*.json")
    base = load(f"results/e1a_tailfill/raw_base/linksign/{ds}/*.json")
    cells = []
    for k in LINK:
        if 42 in test and 42 in base:
            cells.append(f"{(test[42][k]-base[42][k])*1000:+.1f}")
        else:
            cells.append("—")
    ap_(f"{ds:<22}" + "".join(f"{c:>12}" for c in cells))
ap_("")

ap_("== B. B2（sign seed42） vs full ==")
ap_(f"{'ds':<22}" + "".join(f"{k:>12}" for k in SIGN))
for ds in DS:
    test = load(f"results/bte_stage1/raw/sign/{ds}/*.json")
    base = load(f"results/sign_valthr/raw/{ds}/SignDyGFormer_seed*.json")
    cells = []
    for k in SIGN:
        if 42 in test and 42 in base:
            cells.append(f"{(test[42][k]-base[42][k])*1000:+.1f}")
        else:
            cells.append("—")
    ap_(f"{ds:<22}" + "".join(f"{c:>12}" for c in cells))
ap_("")

ap_("== C. sign w/o BTE（seed42） vs full ==")
ap_(f"{'ds':<22}" + "".join(f"{k:>12}" for k in SIGN))
for ds in DS:
    test = load(f"results/sign_nobte/raw/{ds}/*.json")
    base = load(f"results/sign_valthr/raw/{ds}/SignDyGFormer_seed*.json")
    cells = []
    for k in SIGN:
        if 42 in test and 42 in base:
            cells.append(f"{(test[42][k]-base[42][k])*1000:+.1f}")
        else:
            cells.append("—")
    ap_(f"{ds:<22}" + "".join(f"{c:>12}" for c in cells))
ap_("")
ap_("注：单种子屏幕（seed42）；方向为正的数据集才进入 5 种子确认段。")

out = ROOT / "results/bte_screen_verdict_20260925.txt"
out.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"写出 {out}")
print("\n".join(L))
