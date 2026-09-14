# -*- coding: utf-8 -*-
"""nh5 补充集指标汇总（拉取后打印）：胜者/复核 5 种子 mean±std、邻域单点、探边 + 主表现行对照（含配对 t）。"""
import glob
import json
import pathlib
import re
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


def by_seed(paths):
    out = {}
    for p in paths:
        m = re.search(r"seed(\d+)", pathlib.Path(p).name)
        if m:
            out[m.group(1)] = load(p)["auc"]
    return out


def paired(title, cand_glob, cur_glob):
    """同 seed 配对比较（候选 vs 现行）：Δmean、配对 t（df=n-1；df=4 时 |t|>2.78 ⇔ p<0.05）。"""
    a, b = by_seed(glob.glob(cand_glob)), by_seed(glob.glob(cur_glob))
    seeds = sorted(set(a) & set(b), key=int)
    diffs = [a[s] - b[s] for s in seeds]
    n = len(diffs)
    md = sum(diffs) / n
    sd = (sum((d - md) ** 2 for d in diffs) / (n - 1)) ** 0.5 if n > 1 else float("nan")
    t = md / (sd / n ** 0.5) if sd and sd > 0 else float("nan")
    print(f"== 配对比较 · {title} ==")
    for s in seeds:
        print(f"  seed{s:<6s} cand={a[s]:.4f} cur={b[s]:.4f} Δ={a[s] - b[s]:+.4f}")
    print(f"  → n={n} Δmean={md:+.4f} sd(Δ)={sd:.4f} t={t:.2f}（df={n - 1}）")
    print()

print("################ nh5 新产物 ################")
show("RT 胜者 NN-100/LF-3 ×5种子", glob.glob(str(RAW / "RedditHyperlinkTitle" / "*NN-100.LF-3.*.json")))
show("RB 胜者 NN-40/LF-1 ×5种子", glob.glob(str(RAW / "RedditHyperlinkBody" / "*NN-40.LF-1.*.json")))
show("WV 复核 NN-20/LF-20 ×5种子（#61）", glob.glob(str(RAW / "WikiVote" / "*NN-20.LF-20.*.json")))
show("OTC 复核 NN-40/LF-15 ×5种子（#62）", glob.glob(str(RAW / "BitcoinOTC" / "*NN-40.LF-15.*.json")))
show("OTC 邻域（已出）", glob.glob(str(RAW / "BitcoinOTC" / "*.json")))
show("WV 邻域（已出）", glob.glob(str(RAW / "WikiVote" / "*.json")))
show("RT 探边", glob.glob(str(RAW / "RedditHyperlinkTitle" / "*NN-100.LF-5.*.json")) + glob.glob(str(RAW / "RedditHyperlinkTitle" / "*NN-120.LF-3.*.json")))

print("################ 对照：主表 sign 现行配置（5 种子） ################")
show("RT 现行 NN-100/LF-1 ×5", glob.glob(str(MAIN / "RedditHyperlinkTitle" / "*NN-100.LF-1.*.json")))
show("RB 现行 NN-60/LF-1 ×5", glob.glob(str(MAIN / "RedditHyperlinkBody" / "*NN-60.LF-1.*.json")))
show("OTC 现行 NN-60/LF-10 ×5", glob.glob(str(MAIN / "BitcoinOTC" / "*NN-60.LF-10*.json")))
show("WV 现行 NN-40/LF-15 ×5", glob.glob(str(MAIN / "WikiVote" / "*NN-40.LF-15*.json")))

print("################ 配对比较（候选 vs 现行，同 seed） ################")
paired("WV：NN-20/LF-20（复核） vs NN-40/LF-15（现行）",
       str(RAW / "WikiVote" / "*NN-20.LF-20.*.json"), str(MAIN / "WikiVote" / "*NN-40.LF-15*.json"))
paired("OTC：NN-40/LF-15（复核） vs NN-60/LF-10（现行）",
       str(RAW / "BitcoinOTC" / "*NN-40.LF-15.*.json"), str(MAIN / "BitcoinOTC" / "*NN-60.LF-10*.json"))
paired("RT：NN-100/LF-3（胜者） vs NN-100/LF-1（现行）",
       str(RAW / "RedditHyperlinkTitle" / "*NN-100.LF-3.*.json"), str(MAIN / "RedditHyperlinkTitle" / "*NN-100.LF-1.*.json"))
paired("RB：NN-40/LF-1（胜者） vs NN-60/LF-1（现行）",
       str(RAW / "RedditHyperlinkBody" / "*NN-40.LF-1.*.json"), str(MAIN / "RedditHyperlinkBody" / "*NN-60.LF-1.*.json"))
