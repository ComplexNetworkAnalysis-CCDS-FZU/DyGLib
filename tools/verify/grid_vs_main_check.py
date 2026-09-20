# -*- coding: utf-8 -*-
"""网格 vs 主表 口径 QA：同 (NN,LF) 点上 sign 网格重跑值 vs 主表 full 运行值。

用途：k×N 全量重跑（方案甲）交付前的协议一致性核查——
若同名配置数值不一致，说明网格运行与主表运行存在协议差（如 batch-size 等
per-dataset 调参未随网格传入），需在交付结论中说明。

用法（仓库根）：python tools/verify/grid_vs_main_check.py
"""
import glob
import json
import os
import re

DS_LIST = ["RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha", "BitcoinOTC", "WikiVote"]
NAME_RE = re.compile(r"NN-(\d+)\.LF-(\d+)")


def load_metrics(path):
    try:
        return json.load(open(path, encoding="utf-8")).get("test metrics", {})
    except Exception:
        return {}


def main():
    print("ds | (NN,LF) | grid: auc/f1_macro | main: auc/f1_macro | Δauc/Δf1m")
    print("-" * 100)
    for ds in DS_LIST:
        mains = sorted(glob.glob(f"results/sign_valthr/raw/{ds}/*seed42*.json"))
        if not mains:
            print(f"{ds} | 主表文件缺失")
            continue
        mname = os.path.basename(mains[0])
        mm = NAME_RE.search(mname)
        main_m = load_metrics(mains[0])
        # 主表 run 的实际 NN/LF
        nn, lf = (mm.group(1), mm.group(2)) if mm else ("?", "?")
        grids = glob.glob(f"results/sign_param/raw/{ds}/" + (f"*NN-{nn}.LF-{lf}.*.json" if mm else "*.json"))
        if not grids:
            print(f"{ds} | ({nn},{lf}) | grid 文件缺失（该组合可能不在网格内）")
            continue
        grid_m = load_metrics(grids[0])
        ga, gm_ = float(grid_m.get("auc", "nan")), float(grid_m.get("f1_macro", "nan"))
        ma, mm_ = float(main_m.get("auc", "nan")), float(main_m.get("f1_macro", "nan"))
        print(f"{ds} | ({nn},{lf}) | {ga:.4f}/{gm_:.4f} | {ma:.4f}/{mm_:.4f} | {ga-ma:+.4f}/{gm_-mm_:+.4f}")


if __name__ == "__main__":
    main()
