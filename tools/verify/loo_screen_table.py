"""loo_screen_table.py — LOO 单种子初筛表（#134/135；seed42）。

对照：full（RAS-E.RASE-E.BTE-E.CNAS-E） vs 四个 LOO 掩码（seed42、NN-Best.LF-Best）：
- w/o CNAS = [T,T,T,F]（CNAS-D）；w/o BTE = [T,T,F,T]（BTE-D）；
- w/o RAE  = [T,F,T,T]（RASE-D）；w/o RAS = [F,T,T,T]（RAS-D）。

输出：每数据集一张表（auc / f1_mac / f1_wt；Δ = mask − full；贡献 = full − mask，正=该模块有帮助）；
末段按掩码汇总 5 数据集贡献（含 |Δauc|≥0.005 计数——扩种子判据参考；seed42 单点、噪声量级 ~±0.003–0.01）。

用法（仓库根）：python tools/verify/loo_screen_table.py [glob]
默认 glob：results/E-2_ablation/raw_seeds/*
"""
from __future__ import annotations

import glob
import json
import pathlib
import sys

KEYS = ["auc", "f1_mac", "f1_wt"]
FULL = "RAS-E.RASE-E.BTE-E.CNAS-E"
MASKS = [
    ("w/o CNAS", "RAS-E.RASE-E.BTE-E.CNAS-D"),
    ("w/o BTE", "RAS-E.RASE-E.BTE-D.CNAS-E"),
    ("w/o RAE", "RAS-E.RASE-D.BTE-E.CNAS-E"),
    ("w/o RAS", "RAS-D.RASE-E.BTE-E.CNAS-E"),
]


def load_s42(d: pathlib.Path, flags: str):
    f = d / f"SignDyGFormer_seed42.NN-Best.LF-Best.{flags}.P1.TE.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8")).get("test metrics", {})


def main() -> None:
    globs = sys.argv[1:] or ["results/E-2_ablation/raw_seeds/*"]
    dirs = []
    for g in globs:
        dirs += [pathlib.Path(p) for p in glob.glob(g) if pathlib.Path(p).is_dir()]
    dirs = [d for d in dirs if load_s42(d, FULL) is not None]
    if not dirs:
        print("[空] 未找到含 full 结果的数据集目录")
        return

    contrib = {m: {"auc": [], "f1_mac": []} for m, _ in MASKS}
    for d in sorted(dirs):
        full = load_s42(d, FULL)
        print(f"== {d.name} ==")
        w = 10
        head = f"{'config':<10s} " + " ".join(f"{k:>{w}s}" for k in KEYS) + "  贡献(full−mask): auc / f1_mac"
        print(head)
        print("-" * len(head))
        print(f"{'full':<10s} " + " ".join(f"{float(full[k]):>{w}.4f}" for k in KEYS))
        for name, flags in MASKS:
            tm = load_s42(d, flags)
            if tm is None:
                print(f"{name:<10s} [缺档]")
                continue
            cells = " ".join(f"{float(tm[k]):>{w}.4f}" for k in KEYS)
            d_auc = float(full["auc"]) - float(tm["auc"])
            d_mac = float(full["f1_mac"]) - float(tm["f1_mac"])
            contrib[name]["auc"].append(d_auc)
            contrib[name]["f1_mac"].append(d_mac)
            flag = "**" if abs(d_auc) >= 0.005 else "  "
            print(f"{name:<10s} {cells}  {flag}+{d_auc:.4f} / {d_mac:+.4f}   (Δauc mask−full = {-d_auc:+.4f})")
        print()

    print("== 汇总：贡献 = full − full∖{m}（seed42）；正 = 去掉该模块变差 = 模块有帮助 ==")
    print(f"{'mask':<10s} {'meanΔauc':>9s} {'pos/neg':>8s} {'|Δ|≥0.005':>10s} {'meanΔf1_mac':>12s}")
    for name, _ in MASKS:
        va, vm = contrib[name]["auc"], contrib[name]["f1_mac"]
        if not va:
            continue
        pos = sum(1 for x in va if x > 0)
        big = sum(1 for x in va if abs(x) >= 0.005)
        print(f"{name:<10s} {sum(va)/len(va):>+9.4f} {pos:>4d}/{len(va)-pos:<3d} {big:>10d} {sum(vm)/len(vm):>+12.4f}")


if __name__ == "__main__":
    main()
