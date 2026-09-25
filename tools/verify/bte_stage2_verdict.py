# -*- coding: utf-8 -*-
"""BTE 阶段二判定（linksign，5 种子配对；B2/B4/B5 vs full）。

test = results/bte_stage1/raw/linksign/{ds}/*.{tag}.json（阶段二写入同名文件，5 种子）
base = results/e1a_tailfill/raw_base/linksign/{ds}/*.json
输出：results/bte_stage2_verdict_20260925.txt（幂等覆盖；可随批更新）
用法：python tools/verify/bte_stage2_verdict.py [--tags B2 B4 B5]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ["WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha", "BitcoinOTC"]
METRICS = ["f1_wt", "f1_mac", "sign_f1", "ap", "auc"]


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


def stat(a, b, k):
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


ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", default=["B2", "B4", "B5"])
args = ap.parse_args()

L = []
say = L.append
say("BTE 阶段二判定（linksign；5 种子配对；Δ‰ 正 = 变体更高）")
say("")
for tag in args.tags:
    say(f"===== {tag} =====")
    say(f"{'ds':<22}" + "".join(f"{k:>22}" for k in METRICS))
    for ds in DS:
        test = load(f"results/bte_stage1/raw/linksign/{ds}/*.{tag}.json")
        base = load(f"results/e1a_tailfill/raw_base/linksign/{ds}/*.json")
        cells = []
        for k in METRICS:
            r = stat(test, base, k)
            if r is None:
                cells.append("—")
            else:
                pstr = f" p={r[3]:.3f}" if r[3] is not None else ""
                cells.append(f"{r[0]:+.1f} ({r[1]}/{r[2]}){pstr}")
        say(f"{ds:<22}" + "".join(f"{c:>22}" for c in cells))
    say("")

# 附：B2-RB 预登记行
try:
    test = load("results/bte_stage1/raw/linksign/RedditHyperlinkBody/*.B2.json")
    base = load("results/e1a_tailfill/raw_base/linksign/RedditHyperlinkBody/*.json")
    r = stat(test, base, "f1_wt")
    if r:
        ok = (r[0] > 0) and (r[3] is not None and r[3] < 0.05 or r[1] == r[2])
        say(f"[预登记] B2-RB f1_wt = {r[0]:+.1f}‰ ({r[1]}/{r[2]}) p={r[3] if r[3] is not None else float('nan'):.4f} "
            f"⇒ {'RB 主指标修复成立（待双口径/复算）' if ok else '未达 A 档判据'}")
except Exception as e:  # noqa: BLE001
    say(f"[预登记] B2-RB 数据未齐：{e}")

out = ROOT / "results/bte_stage2_verdict_20260925.txt"
out.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"写出 {out}")
print("\n".join(L))
