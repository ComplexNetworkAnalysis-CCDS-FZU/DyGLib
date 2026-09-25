# -*- coding: utf-8 -*-
"""T14b：7 方法 × 2 任务 × 5 数据集 × **全指标**切片（同协议重建两表用）。

输出：results/t14b_full_keys_20260925.csv（长表：task, method, dataset, metric, mean, std, n）
     results/t14b_full_keys_20260925.txt（各 task 的指标清单 + 行数摘要）
用法：python tools/verify/t14b_full_keys.py
"""
from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DS = ["WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha", "BitcoinOTC"]
BASELINES = ["gcn", "sgcn", "sigat", "tgn", "semba"]


def _seed(p):
    return int(re.search(r"seed(\d+)", Path(p).name).group(1))


def load(pattern, key="test metrics"):
    out = {}
    for p in glob.glob(str(ROOT / pattern)):
        if "-profiler" in p:
            continue
        j = json.load(open(p, encoding="utf-8"))
        tm = j.get(key, j.get("metrics", {}))
        out[_seed(p)] = tm
    return out


def stat(vals):
    nums = []
    for v in vals:
        if v is None:
            continue
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            continue
    a = np.array(nums, dtype=float)
    if a.size == 0:
        return None, None, 0
    return float(a.mean()), float(a.std(ddof=0)), int(a.size)


rows = []
keys_seen = {"linksign": set(), "sign": set()}
for task in ("linksign", "sign"):
    for method in BASELINES:
        for ds in DS:
            per = load(f"results/semba_ab/raw_full/{method}/{ds}_{task}_seed*.json")
            allk = set()
            for m in per.values():
                allk |= set(m.keys())
            for k in sorted(allk):
                mu, sd, n = stat([m.get(k) for m in per.values()])
                if n:
                    rows.append((task, method, ds, k, mu, sd, n))
                    keys_seen[task].add(k)
    for method, pat in (("DyGFormer", "results/s1_refresh/raw/{t}/{ds}/DyGFormer_seed*.json"),
                        ("ours", "results/e1a_tailfill/raw_base/linksign/{ds}/*.json" if task == "linksign"
                         else "results/sign_valthr/raw/{ds}/SignDyGFormer_seed*.json")):
        for ds in DS:
            per = load(pat.format(t=task, ds=ds))
            allk = set()
            for m in per.values():
                allk |= set(m.keys())
            for k in sorted(allk):
                mu, sd, n = stat([m.get(k) for m in per.values()])
                if n:
                    rows.append((task, method, ds, k, mu, sd, n))
                    keys_seen[task].add(k)

out_csv = ROOT / "results/t14b_full_keys_20260925.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["task", "method", "dataset", "metric", "mean", "std", "n"])
    for r in rows:
        w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.6f}", f"{r[5]:.6f}", r[6]])

L = []
ap_ = L.append
ap_("T14b 全键切片（7 方法 × 2 任务 × 5 数据集 × 全指标；同协议）")
ap_(f"总行数：{len(rows)}")
for task in ("linksign", "sign"):
    ap_(f"[{task}] 指标键（{len(keys_seen[task])}）：{', '.join(sorted(keys_seen[task]))}")
ap_("")
ap_("覆盖核对（apps）：")
for task in ("linksign", "sign"):
    ap_(f"-- {task}")
    ap_(f"{'method':<10}{'ds 数':>8}{'指标数':>8}")
    for method in BASELINES + ["DyGFormer", "ours"]:
        dss = {r[2] for r in rows if r[0] == task and r[1] == method}
        ks = {r[3] for r in rows if r[0] == task and r[1] == method}
        ap_(f"{method:<10}{len(dss):>8}{len(ks):>8}")

out_txt = ROOT / "results/t14b_full_keys_20260925.txt"
out_txt.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"写出 {out_csv}（{len(rows)} 行）")
print(f"写出 {out_txt}")
print("\n".join(L[:8]))
