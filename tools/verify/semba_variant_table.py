# -*- coding: utf-8 -*-
"""SEMBA 逐变体表（Paper 0137 §二：Avg.Rank 重算所需）。

输出：
- results/semba_variant_table_20260924.txt —— 人读表（sign/linksign × 5 变体 × 5 数据集，auc + 主指标 + f1_bin/f1_mac）
- results/semba_variant_table_20260924.csv —— 机读（含 ours / 真 DyG 行；供 Avg.Rank 重算）
指标映射：sign 主=f1_macro；linksign 主=f1_wt ← SEMBA f1_weighted。
缺档：semba RT-linksign 为 4 种子（seed1024 OOM，标 N/A）。
用法：python tools/verify/semba_variant_table.py
"""
from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha", "BitcoinOTC"]
VARIANTS = ["gcn", "sgcn", "sigat", "tgn", "semba"]
SEEDS = [42, 123, 456, 789, 1024]


def load_semba(variant: str, ds: str, task: str) -> dict[int, dict]:
    out = {}
    for p in glob.glob(str(ROOT / f"results/semba_ab/raw_full/{variant}/{ds}_{task}_seed*.json")):
        m = re.search(r"seed(\d+)", Path(p).name)
        out[int(m.group(1))] = json.load(open(p, encoding="utf-8")).get("metrics", {})
    return out


def load_ours(task: str, ds: str) -> dict[int, dict]:
    if task == "sign":
        pat = f"results/sign_valthr/raw/{ds}/SignDyGFormer_seed*.json"
    else:
        pat = f"results/e1a_tailfill/raw_base/linksign/{ds}/*.json"
    out = {}
    for p in glob.glob(str(ROOT / pat)):
        if "-profiler" in p:
            continue
        m = re.search(r"seed(\d+)", Path(p).name)
        tm = json.load(open(p, encoding="utf-8"))["test metrics"]
        out[int(m.group(1))] = {
            "auc": tm.get("auc"), "main": tm.get("f1_macro") if task == "sign" else tm.get("f1_wt"),
            "f1_binary": tm.get("f1_binary"), "f1_macro": tm.get("f1_macro"),
        }
    return out


def load_dyg(task: str, ds: str) -> dict[int, dict]:
    out = {}
    for p in glob.glob(str(ROOT / f"results/s1_refresh/raw/{task}/{ds}/DyGFormer_seed*.json")):
        if "-profiler" in p:
            continue
        m = re.search(r"seed(\d+)", Path(p).name)
        tm = json.load(open(p, encoding="utf-8"))["test metrics"]
        out[int(m.group(1))] = {
            "auc": tm.get("auc"), "main": tm.get("f1_macro") if task == "sign" else tm.get("f1_wt"),
            "f1_binary": tm.get("f1_binary"), "f1_macro": tm.get("f1_macro"),
        }
    return out


def norm_semba(metrics: dict, task: str) -> dict:
    return {
        "auc": metrics.get("auc"),
        "main": metrics.get("f1_macro") if task == "sign" else metrics.get("f1_weighted"),
        "f1_binary": metrics.get("f1_binary"),
        "f1_macro": metrics.get("f1_macro"),
    }


def stat(vals):
    a = np.array([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return None, None, 0
    return float(a.mean()), float(a.std(ddof=0)), int(a.size)


def main() -> int:
    lines, csv_rows = [], []
    for task, main_name in (("sign", "f1_macro"), ("linksign", "f1_wt")):
        lines.append("=" * 130)
        lines.append(f"任务 {task}（主指标 = {main_name}；mean±pstd，5 种子；缺档注明 n）  [表值 x1000 为 ‰ 见正文]")
        lines.append("=" * 130)
        for metric in ("auc", "main", "f1_binary", "f1_macro"):
            if metric == "f1_binary" and task == "linksign":
                label = "f1_bin(映射=f1_binary)"
            elif metric == "main":
                label = f"主指标({main_name})"
            else:
                label = metric
            lines.append(f"\n--- {label} ---")
            header = f"{'variant':<10}" + "".join(f"{ds:<24}" for ds in DATASETS)
            lines.append(header)
            for variant in VARIANTS + ["ours", "DyGFormer"]:
                row = f"{variant:<10}"
                for ds in DATASETS:
                    if variant == "ours":
                        runs = load_ours(task, ds)
                        vals = {s: norm_semba(v, task) for s, v in runs.items()}
                    elif variant == "DyGFormer":
                        runs = load_dyg(task, ds)
                        vals = {s: norm_semba(v, task) for s, v in runs.items()}
                    else:
                        runs = load_semba(variant, ds, task)
                        vals = {s: norm_semba(v, task) for s, v in runs.items()}
                    key = "auc" if metric == "auc" else metric
                    mean, std, n = stat([v.get(key) for v in vals.values()])
                    if mean is None:
                        row += f"{'N/A':<24}"
                    else:
                        tag = "" if n == 5 else f"(n={n})"
                        row += f"{mean:.4f}±{std:.4f}{tag:<6}"[:24]
                        csv_rows.append({
                            "task": task, "variant": variant, "dataset": ds, "metric": metric,
                            "mean": f"{mean:.6f}", "std": f"{std:.6f}", "n": n,
                        })
                lines.append(row)
        # ours / DyG 的 CSV 行也补全（固定 5 种子）
        for variant, loader in (("ours", load_ours), ("DyGFormer", load_dyg)):
            for ds in DATASETS:
                runs = {s: norm_semba(v, task) for s, v in loader(task, ds).items()}
                for metric in ("auc", "main", "f1_binary", "f1_macro"):
                    mean, std, n = stat([v.get(metric) for v in runs.values()])
                    if mean is not None:
                        csv_rows.append({
                            "task": task, "variant": variant, "dataset": ds, "metric": metric,
                            "mean": f"{mean:.6f}", "std": f"{std:.6f}", "n": n,
                        })

    txt = "\n".join(lines) + "\n"
    (ROOT / "results/semba_variant_table_20260924.txt").write_text(txt, encoding="utf-8")
    with open(ROOT / "results/semba_variant_table_20260924.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "variant", "dataset", "metric", "mean", "std", "n"])
        w.writeheader()
        w.writerows(csv_rows)
    print(txt)
    print(f"[ok] CSV: results/semba_variant_table_20260924.csv（{len(csv_rows)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
