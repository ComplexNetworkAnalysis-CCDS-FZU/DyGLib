# -*- coding: utf-8 -*-
"""网格 vs 主表 口径 QA（v2，2026-09-28）：新网格（grid_new）在当前配置点 vs 主表运行值。

说明：v1 误读旧 sign_param 批（09-24）——已改读 results/grid_new/raw。
- sign：主表 = results/sign_valthr/raw/{ds}/*seed42*.json；网格 = grid_new/raw/sign/{ds} 同名格
- linksign：主表 = results/e1a_tailfill/raw_base/linksign/{ds}/*seed42*.json；网格 = grid_new/raw/linksign/{ds}
差值 = 单种子下同配置的再现性 + 协议差（若有）；用于交付结论中的口径说明。

用法（仓库根）：python tools/verify/grid_vs_main_check.py
"""
import glob
import json
import os
import re

NAME_RE = re.compile(r"NN-(\d+)\.LF-(\d+)")


def load_metrics(path):
    try:
        return json.load(open(path, encoding="utf-8")).get("test metrics", {})
    except Exception:
        return {}


def key(m):
    if "f1_mac" in m and "sign_f1" in m:
        return ("auc", "f1_mac")
    if "f1_macro" in m:
        return ("auc", "f1_macro")
    return ("auc", "f1_mac")


def main():
    print("task | ds | (NN,LF) | grid: auc/f1_mac | main: auc/f1_mac | Δauc/Δf1m  [grid file]")
    print("-" * 108)
    cases = [
        ("sign", "results/sign_valthr/raw", "results/grid_new/raw/sign"),
        ("linksign", "results/e1a_tailfill/raw_base/linksign", "results/grid_new/raw/linksign"),
    ]
    for task, main_dir, grid_dir in cases:
        for ds in ["RedditHyperlinkTitle", "RedditHyperlinkBody", "WikiVote"]:
            mains = sorted(glob.glob(f"{main_dir}/{ds}/*seed42*.json"))
            if not mains:
                print(f"{task} | {ds} | 主表文件缺失")
                continue
            mm = NAME_RE.search(os.path.basename(mains[0]))
            if not mm:
                print(f"{task} | {ds} | 主表名无 NN/LF")
                continue
            nn, lf = mm.group(1), mm.group(2)
            grids = glob.glob(f"{grid_dir}/{ds}/*NN-{nn}.LF-{lf}.*.json")
            if not grids:
                print(f"{task} | {ds} | ({nn},{lf}) | grid 文件缺失")
                continue
            gm = load_metrics(grids[0])
            mm_ = load_metrics(mains[0])
            ka, kf = key(gm)
            ga, gf = float(gm.get(ka, "nan")), float(gm.get(kf, "nan"))
            ma, mf = float(mm_.get(ka, "nan")), float(mm_.get(kf, "nan"))
            print(f"{task:8s} | {ds:<22s} | ({nn},{lf}) | {ga:.4f}/{gf:.4f} | {ma:.4f}/{mf:.4f} | "
                  f"{ga-ma:+.4f}/{gf-mf:+.4f}  [{os.path.basename(grids[0])[:52]}]")


if __name__ == "__main__":
    main()

