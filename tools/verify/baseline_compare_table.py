# -*- coding: utf-8 -*-
"""基线对照总表（2026-09-23）：ours vs DyGFormer vs SEMBA 五变体（+DySDGNN/DynamiSE 备注）。

数据源（本地归档）：
- ours sign（5×5）        results/sign_valthr/raw/{ds}/SignDyGFormer_seed*.json     指标 {auc, f1_macro}
- ours linksign（5×5）    results/e1a_tailfill/raw_base/linksign/{ds}/*.json       指标 {auc, f1_wt}
- DyGFormer               results/s1_refresh/raw/{task}/{ds}/DyGFormer_seed*.json
- SEMBA 五变体            results/semba_ab/raw_full/{variant}/{ds}_{task}_seed*.json 指标 {auc, f1_macro|f1_weighted}
- DySDGNN/DynamiSE        results/baseline_m5/raw/{model}/{ds}_seed*.json           指标 {AUC, F1_bin}（另一任务口径，仅备注）

用法（仓库根；需 numpy）：python tools/verify/baseline_compare_table.py
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha", "BitcoinOTC"]
VARIANTS = ["gcn", "sgcn", "sigat", "tgn", "semba"]
SEEDS = [42, 123, 456, 789, 1024]


def load_glob(pattern: str, metric_keys: list[str]) -> dict[int, dict[str, float]]:
    out = {}
    for p in glob.glob(str(ROOT / pattern)):
        if "-profiler" in p:
            continue
        m = re.search(r"seed(\d+)", Path(p).name)
        if not m:
            continue
        d = json.load(open(p, encoding="utf-8"))
        tm = d.get("test metrics") or d.get("metrics") or {}
        out[int(m.group(1))] = {k: float(tm[k]) for k in metric_keys if k in tm}
    return out


def stat(runs: dict, key: str):
    vals = [v[key] for v in runs.values() if key in v]
    if not vals:
        return None, None, 0
    a = np.array(vals)
    return float(a.mean()), float(a.std(ddof=0)), len(a)


def fmt(mean, std, n):
    return "—" if mean is None else f"{mean:.4f}±{std:.4f}(n={n})"


def run_table(task: str, ours_metric: str, semba_metric: str):
    print("=" * 118)
    print(f"任务 = {task}；主指标列 = {ours_metric}（SEMBA 对应 {semba_metric}）")
    print("=" * 118)
    for metric_label, ours_key, dyg_key, semba_key in (
        ("auc", "auc", "auc", "auc"),
        (f"主指标 {ours_metric}", ours_metric, ours_metric, semba_metric),
    ):
        print(f"\n--- {metric_label}（5 种子 mean±pstd）---")
        head = f"{'数据集':<22}{'ours':<26}{'DyGFormer':<26}"
        head += "".join(f"{v:<26}" for v in VARIANTS) + "最佳SEMBA"
        print(head)
        for ds in DATASETS:
            if task == "sign":
                ours = load_glob(f"results/sign_valthr/raw/{ds}/SignDyGFormer_seed*.json", [ours_key])
            else:
                ours = load_glob(f"results/e1a_tailfill/raw_base/linksign/{ds}/*.json", [ours_key])
            dyg = load_glob(f"results/s1_refresh/raw/{task}/{ds}/DyGFormer_seed*.json", [dyg_key])
            row = f"{ds:<22}{fmt(*stat(ours, ours_key)):<26}{fmt(*stat(dyg, dyg_key)):<26}"
            best_name, best_v = None, None
            for var in VARIANTS:
                runs = {}
                for p in glob.glob(str(ROOT / f"results/semba_ab/raw_full/{var}/{ds}_{task}_seed*.json")):
                    m = re.search(r"seed(\d+)", Path(p).name)
                    d = json.load(open(p, encoding="utf-8"))
                    tm = d.get("metrics") or {}
                    runs[int(m.group(1))] = {k: float(tm[k]) for k in (ours_key, semba_key) if k in tm}
                mean, std, n = stat(runs, semba_key)
                row += f"{fmt(mean, std, n):<26}"
                if mean is not None and (best_v is None or mean > best_v):
                    best_name, best_v = var, mean
            ours_mean = stat(ours, ours_key)[0]
            delta = "" if ours_mean is None or best_v is None else f"{ours_mean - best_v:+.4f}"
            print(row + f"{best_name}({best_v:.4f}) Δours={delta}")
    print()


def main() -> int:
    run_table("sign", "f1_macro", "f1_macro")
    run_table("linksign", "f1_wt", "f1_weighted")

    print("=" * 118)
    print("备注：DySDGNN / DynamiSE（results/baseline_m5；任务口径以 Baseline 定义为准；指标 {AUC, F1_bin}）")
    print("=" * 118)
    for model in ["DySDGNN", "DynamiSE"]:
        for ds in DATASETS:
            runs = load_glob(f"results/baseline_m5/raw/{model}/{ds}_seed*.json", ["AUC", "F1_bin"])
            mean, std, n = stat(runs, "AUC")
            if mean is None:
                print(f"{model:<10}{ds:<22}（缺档）")
                continue
            f1m, f1s, _ = stat(runs, "F1_bin")
            print(f"{model:<10}{ds:<22}AUC={mean:.4f}±{std:.4f}(n={n})  F1_bin={f1m:.4f}±{f1s:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
