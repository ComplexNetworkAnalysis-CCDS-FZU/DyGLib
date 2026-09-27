# -*- coding: utf-8 -*-
"""5 个公共基线（gcn/sgcn/sigat/tgn/semba）的 f1_mac 并列列切片（Paper 300e §二.3）。

说明：
- 基线原件键 = `f1_macro`（对应 ours/DyG 的 `f1_mac` 列；sign 任务同一键名）；
- 数据源：results/semba_ab/raw_full/{variant}/{ds}_{task}_seed*.json（5 种子）；
- 附 ours / DyGFormer 对照行：
    ours: linksign → results/e1a_tailfill/raw_base/linksign/{ds}；sign → results/sign_valthr/raw/{ds}
    DyG:  linksign → results/s1_refresh/raw/linksign/{ds}；sign → RT/RB 用 raw_valthr，其余用 raw
输出：results/baseline5_f1mac_tables_20260927.txt / .csv
用法：python tools/verify/baseline5_f1mac_slice.py
"""
import csv
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ["WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha", "BitcoinOTC"]
BASELINES = ["gcn", "sgcn", "sigat", "tgn", "semba"]


def load(pattern):
    out = []
    for p in sorted(glob.glob(str(ROOT / pattern))):
        if "-profiler" in p:
            continue
        j = json.load(open(p, encoding="utf-8"))
        out.append(j.get("metrics", j.get("test metrics", {})))
    return out


def get(per, task):
    key = "f1_macro" if task in ("linksign", "sign") else "f1_mac"
    vals = []
    for m in per:
        v = m.get(key, m.get("f1_mac") if key == "f1_macro" else None)
        if v is None:
            v = m.get("f1_macro") if task in ("linksign", "sign") else m.get("f1_mac")
        if v is not None:
            vals.append(float(v))
    return vals


def cell(vals):
    if not vals:
        return "—"
    mu = sum(vals) / len(vals)
    sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
    return f"{mu:.4f}±{sd:.4f}(n={len(vals)})"


METHODS = []  # (label, task, ds) -> 由 loader 提供

lines, csv_rows = [], []


def say(s=""):
    lines.append(s)
    print(s)


say("5 个公共基线 · f1_mac 并列列（键=f1_macro；5 种子 mean±std）")
say("（附 ours / DyGFormer 对照；DyG sign RT/RB 取 val-thr 正典 raw_valthr）")
say()
for task in ("linksign", "sign"):
    say(f"===== {task} · f1_mac =====")
    hdr = f"{'dataset':<22s}" + "".join(f"{m:>22s}" for m in BASELINES + ["DyGFormer", "ours"])
    say(hdr)
    for ds in DS:
        row = f"{ds:<22s}"
        for v in BASELINES:
            vals = get(load(f"results/semba_ab/raw_full/{v}/{ds}_{task}_seed*.json"), task)
            row += f"{cell(vals):>22s}"
            csv_rows.append((task, v, ds, "f1_macro", vals))
        # DyG
        if task == "linksign":
            vals = get(load(f"results/s1_refresh/raw/linksign/{ds}/DyGFormer_seed*.json"), task)
        else:
            root = "raw_valthr" if ds in ("RedditHyperlinkTitle", "RedditHyperlinkBody") else "raw"
            vals = get(load(f"results/s1_refresh/{root}/sign/{ds}/DyGFormer_seed*.json"), task)
        row += f"{cell(vals):>22s}"
        csv_rows.append((task, "DyGFormer", ds, "f1_macro", vals))
        # ours
        pat = (f"results/e1a_tailfill/raw_base/linksign/{ds}/*.json" if task == "linksign"
               else f"results/sign_valthr/raw/{ds}/SignDyGFormer_seed*.json")
        vals = get(load(pat), task)
        row += f"{cell(vals):>22s}"
        csv_rows.append((task, "ours", ds, "f1_macro" if task == "sign" else "f1_mac", vals))
        say(row)
    say()

OUT = ROOT / "results/baseline5_f1mac_tables_20260927.txt"
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
with open(ROOT / "results/baseline5_f1mac_tables_20260927.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["task", "method", "dataset", "key", "n", "mean", "std", "values"])
    for task, meth, ds, key, vals in csv_rows:
        if vals:
            mu = sum(vals) / len(vals)
            sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
            w.writerow([task, meth, ds, key, len(vals), f"{mu:.6f}", f"{sd:.6f}", ";".join(f"{v:.6f}" for v in vals)])
print(f"\n[ok] 写出 {OUT.relative_to(ROOT)} 与同名 .csv")
