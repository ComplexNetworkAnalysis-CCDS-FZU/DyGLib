# -*- coding: utf-8 -*-
"""双半径验证批表格：扫描带 `.RLF-{k_r}` 标记的结果 JSON，按数据集输出 k_r 曲线。

用法（仓库根）：
    python tools/verify/ras_radius_table.py                          # 默认 results/ras_radius/raw/*/*
    python tools/verify/ras_radius_table.py "results/ras_radius/raw/RedditHyperlinkTitle/*"
    python tools/verify/ras_radius_table.py --keys auc ap f1_wt f1_mac <globs...>

输出：每个数据集一张表（k_r 升序）；`*=k_c` 标该数据集对角点（k_r == k_c，即原公式）；
`best` 标 auc 最大行；表尾给出「最佳 vs 对角」的 auc 差值。
文件名约定：`...NN-{n}.LF-{k_c}.RLF-{k_r}.RAS-...`（k_r 显式给出时才会带 .RLF-）。
"""
import argparse
import glob
import json
import pathlib
import re

DEFAULT_KEYS = ["auc", "ap", "f1_wt", "f1_mac"]
FNAME_RE = re.compile(r"NN-(\d+)\.LF-(\d+)\.RLF-(\d+)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="*", default=None)
    ap.add_argument("--keys", nargs="+", default=DEFAULT_KEYS)
    args = ap.parse_args()
    globs = args.globs or ["results/ras_radius/raw/*/*"]

    files = []
    for g in globs:
        files += glob.glob(g)

    by_ds = {}
    for f in sorted(files):
        m = FNAME_RE.search(pathlib.Path(f).name)
        if not m:
            continue
        n, kc, kr = (int(x) for x in m.groups())
        d = json.load(open(f, encoding="utf-8")).get("test metrics", {})
        by_ds.setdefault(pathlib.Path(f).parent.name, []).append((kr, kc, n, d))

    if not by_ds:
        print("[空] 没有匹配的 .RLF- 结果文件")
        return

    for ds in sorted(by_ds):
        rows = sorted(by_ds[ds])
        print(f"== {ds} ==")
        w = max(len(k) for k in args.keys)
        header = "k_r".ljust(5) + " | " + " | ".join(k.ljust(w) for k in args.keys) + " | note"
        print(header)
        print("-" * len(header))
        aucs = [(r[3].get("auc"), r[0]) for r in rows]
        aucs = [(float(a), k) for a, k in aucs if isinstance(a, (int, float))]
        best_k = max(aucs)[1] if aucs else None
        diag_auc = None
        for kr, kc, _n, d in rows:
            cells = []
            for k in args.keys:
                v = d.get(k, "-")
                try:
                    cells.append(f"{float(v):.4f}".ljust(w))
                except (TypeError, ValueError):
                    cells.append(str(v).ljust(w))
            note = []
            if kr == kc:
                note.append("*=k_c")
                diag_auc = d.get("auc")
            if kr == best_k:
                note.append("best")
            print(f"{kr:<5}" + " | " + " | ".join(cells) + " | " + " ".join(note))
        if diag_auc is not None and best_k is not None:
            ba = {k: a for a, k in aucs}
            print(
                f"  Δ(best k_r={best_k} − 对角 {diag_auc:.4f}): auc "
                f"{ba.get(best_k, float('nan')) - float(diag_auc):+.4f}"
            )
        print()


if __name__ == "__main__":
    main()
