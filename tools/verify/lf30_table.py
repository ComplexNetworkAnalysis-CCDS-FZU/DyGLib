"""LF 单变量扫描汇总（LF30）：NN 固定为当前配置，逐格对比 LF∈{1,3,5,10,15}。

输入：results/lf30/raw/{linksign,sign}/{ds}/*.json（seed42；2026-09-26 服务器批次）
参照点 L*：RT linksign=1、RB linksign=3、WV linksign=10、RT sign=1、RB sign=1、WV sign=15
主指标：linksign→f1_wt/f1_mac/auc；sign→f1_binary/f1_macro/auc；Δ 相对 L*（‰，正=该点更高）
输出：results/lf30_table_20260926.txt（幂等覆盖）
用法：python tools/verify/lf30_table.py
"""
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/lf30_table_20260926.txt"

COMBOS = [
    # (task, ds, NN, L*)
    ("linksign", "RedditHyperlinkTitle", 60, 1),
    ("linksign", "RedditHyperlinkBody", 80, 3),
    ("linksign", "WikiVote", 15, 10),
    ("sign", "RedditHyperlinkTitle", 100, 1),
    ("sign", "RedditHyperlinkBody", 60, 1),
    ("sign", "WikiVote", 40, 15),
]
LFS = [1, 3, 5, 10, 15]


def load(task, ds):
    rows = {}
    for p in sorted(glob.glob(str(ROOT / f"results/lf30/raw/{task}/{ds}/*.json"))):
        m = re.search(r"LF-(\d+)\.", Path(p).name)
        if not m or int(m.group(1)) not in LFS:
            continue
        rows[int(m.group(1))] = json.load(open(p, encoding="utf-8"))["test metrics"]
    return rows


lines = []


def say(s=""):
    lines.append(s)
    print(s)


def f(v):
    return float(v)


def fm(v, base):
    return f"{(f(v) - f(base)) * 1000:+.1f}"


say("LF 单变量扫描汇总（LF30；seed42；NN=当前配置；2026-09-26 服务器批次）")
say("Δ = 相对当前配置点 L*（‰，正 = 该 LF 点更高）；* = 当前配置点")
say()

for task, ds, nn, lstar in COMBOS:
    rows = load(task, ds)
    n_missing = [lf for lf in LFS if lf not in rows]
    say(f"===== {task} / {ds}  (NN={nn}；L*={lstar}) =====")
    if n_missing:
        say(f"  [缺格] {n_missing}")
    base = rows.get(lstar)
    if base is None:
        say("  [跳过] 参照点缺失")
        say()
        continue
    if task == "linksign":
        cols = [("f1_wt", "f1_wt"), ("f1_mac", "f1_mac"), ("auc", "auc"), ("sign_f1", "[子]sign_f1")]
    else:
        cols = [("f1_binary", "f1_bin"), ("f1_macro", "f1_mac"), ("auc", "auc")]
    hdr = "LF   " + "".join(f"{lbl:>10s}{'Δ‰':>8s}" for _, lbl in cols)
    say(hdr)
    for lf in LFS:
        if lf not in rows:
            say(f"{lf:<4d} (缺)")
            continue
        r = rows[lf]
        mark = "*" if lf == lstar else " "
        cells = ""
        for key, _ in cols:
            if key not in r:
                cells += f"{'--':>10s}{'--':>8s}"
            else:
                cells += f"{f(r[key]):>10.4f}{fm(r[key], base[key]):>8s}"
        say(f"{str(lf) + mark:<4s} {cells}")
    # 方向摘要（每个指标：LF 序 → Δ 序）
    say("  方向摘要：")
    for key, lbl in cols:
        if key not in base:
            continue
        seq = "  ".join(
            f"LF{lf}:{fm(rows[lf][key], base[key])}" if lf in rows and key in rows[lf] else f"LF{lf}:--"
            for lf in LFS
        )
        say(f"    {lbl:<12s} {seq}")
    say()

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[ok] 写出 {OUT.relative_to(ROOT)}")
