# -*- coding: utf-8 -*-
"""T17：7 方法 × {sign_f1 / f1_micro / exist_f1 / ap}（+auc）切片（linksign 主用；sign 附表）。

来源（本地归档）：
  - 5 公共基线变体（gcn/sgcn/sigat/tgn/semba）：results/semba_ab/raw_full/{variant}/{ds}_{task}_seed*.json → "metrics"
  - 真 DyG：results/s1_refresh/raw/{task}/{ds}/DyGFormer_seed*.json → "test metrics"
  - ours：linksign = results/e1a_tailfill/raw_base/linksign/{ds}/*.json；sign = results/sign_valthr/raw/{ds}/SignDyGFormer_seed*.json
输出：results/t17_metric_slice.txt（人读）+ results/t17_metric_slice.csv（机读）
用法：python tools/verify/t17_metric_slice.py（仓库根；本地）
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
METHODS = BASELINES + ["DyGFormer", "ours"]


def _seed(p):
    return int(re.search(r"seed(\d+)", Path(p).name).group(1))


def load_baseline(variant, ds, task):
    out = {}
    for p in glob.glob(str(ROOT / f"results/semba_ab/raw_full/{variant}/{ds}_{task}_seed*.json")):
        out[_seed(p)] = json.load(open(p, encoding="utf-8")).get("metrics", {})
    return out


def load_dyg(ds, task):
    out = {}
    for p in glob.glob(str(ROOT / f"results/s1_refresh/raw/{task}/{ds}/DyGFormer_seed*.json")):
        if "-profiler" in p:
            continue
        out[_seed(p)] = json.load(open(p, encoding="utf-8")).get("test metrics", {})
    return out


def load_ours(ds, task):
    if task == "linksign":
        pat = str(ROOT / f"results/e1a_tailfill/raw_base/linksign/{ds}/*.json")
    else:
        pat = str(ROOT / f"results/sign_valthr/raw/{ds}/SignDyGFormer_seed*.json")
    out = {}
    for p in glob.glob(pat):
        if "-profiler" in p:
            continue
        out[_seed(p)] = json.load(open(p, encoding="utf-8")).get("test metrics", {})
    return out


# 指标映射：键名统一到（sign_f1, f1_micro, exist_f1, ap, auc）
def pick_metrics(m: dict, task: str):
    if task == "linksign":
        return {
            "sign_f1": m.get("sign_f1"),
            "f1_micro": m.get("f1_micro") or m.get("f1_mic"),
            "exist_f1": m.get("exist_f1"),
            "ap": m.get("ap"),
            "auc": m.get("auc"),
        }
    # sign（二分类）：sign_f1/exist_f1 不适用 → None；f1_micro 以 f1_binary 记录对照
    return {
        "sign_f1": None,
        "f1_micro": m.get("f1_binary") or m.get("f1_micro"),
        "exist_f1": None,
        "ap": m.get("ap"),
        "auc": m.get("auc"),
    }


def stat(vals):
    a = np.array([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return None, None, 0
    return float(a.mean()), float(a.std(ddof=0)), int(a.size)


METRICS = ["sign_f1", "f1_micro", "exist_f1", "ap", "auc"]


def main():
    rows = []
    for task in ("linksign", "sign"):
        for method in METHODS:
            for ds in DS:
                if method in BASELINES:
                    per = load_baseline(method, ds, task)
                elif method == "DyGFormer":
                    per = load_dyg(ds, task)
                else:
                    per = load_ours(ds, task)
                vals = {k: [] for k in METRICS}
                for s, m in per.items():
                    pm = pick_metrics(m, task)
                    for k in METRICS:
                        if pm.get(k) is not None:
                            vals[k].append(float(pm[k]))
                for k in METRICS:
                    mu, sd, n = stat(vals[k])
                    rows.append((task, method, ds, k, mu, sd, n))

    out_csv = ROOT / "results/t17_metric_slice.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "method", "dataset", "metric", "mean", "std", "n"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3],
                        "" if r[4] is None else f"{r[4]:.6f}",
                        "" if r[5] is None else f"{r[5]:.6f}", r[6]])

    L = []
    ap_ = L.append
    ap_("T17 指标切片（7 方法；linksign = sign_f1/f1_micro/exist_f1/ap 主角；sign = ap/f1_binary 对照）")
    ap_("来源：semba_ab/raw_full（5 变体）、s1_refresh（DyG）、e1a_tailfill/raw_base + sign_valthr（ours）")
    for task in ("linksign", "sign"):
        ap_("")
        ap_(f"===== {task} =====")
        hdr = f"{'method':<10}{'dataset':<22}" + "".join(f"{k:>10}" for k in METRICS)
        ap_(hdr)
        for method in METHODS:
            for ds in DS:
                cell = []
                for k in METRICS:
                    rr = next(r for r in rows if r[0] == task and r[1] == method and r[2] == ds and r[3] == k)
                    cell.append("—" if rr[4] is None else f"{rr[4]:.4f}")
                ap_(f"{method:<10}{ds:<22}" + "".join(f"{c:>10}" for c in cell))
    out_txt = ROOT / "results/t17_metric_slice.txt"
    out_txt.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"写出 {out_csv}")
    print(f"写出 {out_txt}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
