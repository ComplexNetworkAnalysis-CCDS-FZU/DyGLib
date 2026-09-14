# -*- coding: utf-8 -*-
"""指标速览：打印若干 JSON 结果的紧凑全指标表（用于消融/对照快速比对）。

用法（仓库根目录）：
    python tools/verify/metrics_table.py results/E-2_ablation/raw/BitcoinAlpha/*.json results/E-2_ablation/raw_bte/BitcoinAlpha/*.json
    python tools/verify/metrics_table.py --keys auc ap sign_f1 f1_mac <globs...>   # 自选指标

说明：
  - 结果 JSON 的指标在 "test metrics" 下；缺失显示 "-"。
  - 标签 = 文件名中 `LF-Best.` 之后、`.P1.` 之前的部分（模块旗标）；若无则退化为文件名。
"""
import argparse
import glob
import json
import pathlib

DEFAULT_KEYS = ["auc", "ap", "sign_f1", "f1_mac", "f1_wt", "f1_mic", "acc"]


def tag_of(name: str) -> str:
    if "LF-Best." in name and ".P1." in name:
        return name.split("LF-Best.")[1].split(".P1.")[0]
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="+", help="文件 glob（可多个；shell 会展开）")
    ap.add_argument("--keys", nargs="+", default=DEFAULT_KEYS)
    args = ap.parse_args()

    rows = []
    for g in args.globs:
        for f in sorted(glob.glob(g)):
            d = json.load(open(f, encoding="utf-8")).get("test metrics", {})
            rows.append((pathlib.Path(f).name, tag_of(pathlib.Path(f).name), d))

    if not rows:
        print("[空] 没有匹配文件")
        return

    w = max(len(t) for _, t, _ in rows)
    header = "TAG".ljust(w) + " | " + " | ".join(k.ljust(7) for k in args.keys)
    print(header)
    print("-" * len(header))
    for _, t, d in rows:
        cells = []
        for k in args.keys:
            v = d.get(k, "-")
            try:
                cells.append(f"{float(v):.4f}".ljust(7))
            except (TypeError, ValueError):
                cells.append(str(v).ljust(7))
        print(t.ljust(w) + " | " + " | ".join(cells))


if __name__ == "__main__":
    main()
