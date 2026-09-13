# -*- coding: utf-8 -*-
"""临时：nh5 集指标汇总（拉取后打印；用完即删）。"""
import glob
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load(p):
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    m = d["test metrics"]

    def fl(x):
        try:
            return float(x)
        except Exception:
            return float("nan")

    return {k: fl(m.get(k)) for k in ("auc", "ap", "f1_binary", "f1_weighted")}


def agg(paths):
    rows = []
    for p in sorted(paths):
        m = load(p)
        rows.append((p, m["auc"], m["ap"], m["f1_binary"], m["f1_weighted"]))
    return rows


def show(title, paths):
    print(f"== {title} ==")
    for p, auc, ap, fb, fw in agg(paths):
        print(f"  {pathlib.Path(p).name:<75s} auc={auc:.4f} ap={ap:.4f} f1b={fb:.4f} f1w={fw:.4f}")
    if len(paths) > 1:
        for name, idx in (("auc", 1), ("ap", 2), ("f1b", 3), ("f1w", 4)):
            vals = [r[idx] for r in agg(paths)]
            mean = sum(vals) / len(vals)
            sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
            print(f"  → {name}: mean={mean:.4f} std(ddof=1)={sd:.4f}")
    print()


RAW = ROOT / "results" / "sign_neighborhood" / "raw"
MAIN = ROOT / "results" / "main_tables" / "raw" / "sign"

print("################ nh5 新产物 ################")
show("RT 胜者 NN-100/LF-3 ×5种子", glob.glob(str(RAW / "RedditHyperlinkTitle" / "*NN-100.LF-3.*.json")))
show("RB 胜者 NN-40/LF-1 ×5种子", glob.glob(str(RAW / "RedditHyperlinkBody" / "*NN-40.LF-1.*.json")))
show("OTC 邻域（已出）", glob.glob(str(RAW / "BitcoinOTC" / "*.json")))
show("WV 邻域（已出）", glob.glob(str(RAW / "WikiVote" / "*.json")))
show("RT 探边", glob.glob(str(RAW / "RedditHyperlinkTitle" / "*NN-100.LF-5.*.json")) + glob.glob(str(RAW / "RedditHyperlinkTitle" / "*NN-120.LF-3.*.json")))

print("################ 对照：主表 sign 现行配置（seed42 及 5 种子） ################")
show("RT 现行 NN-100/LF-1 ×5", glob.glob(str(MAIN / "RedditHyperlinkTitle" / "*NN-100.LF-1.*.json")))
show("RB 现行 NN-60/LF-1 ×5", glob.glob(str(MAIN / "RedditHyperlinkBody" / "*NN-60.LF-1.*.json")))
show("OTC 现行 NN-60/LF-10（seed42）", glob.glob(str(MAIN / "BitcoinOTC" / "*seed42*NN-60.LF-10*.json")))
show("WV 现行 NN-40/LF-15（seed42）", glob.glob(str(MAIN / "WikiVote" / "*seed42*NN-40.LF-15*.json")))
