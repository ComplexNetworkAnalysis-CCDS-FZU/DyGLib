"""export_valthr_tables.py — 生成 docs/TABLES_valthr_20260917.md（Paper 补单③ 所需全指标表）。

内容（每数据集：mean±pstd(ddof=0) + 5 种子原值）：
A. 真 DyGFormer（#121–133，val-thr）：sign（RT/RB 为刷新版）+ linksign；
B. ours sign（full 口径，#129–133，val-thr）；
C. ours linksign full（#75–82 批次；供 Paper 校准 f1_mac 口径）。

用法（仓库根）：python tools/verify/export_valthr_tables.py [--out docs/TABLES_valthr_20260917.md]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from valthr_sign_report import (  # noqa: E402
    DATASETS,
    DYG_LINKSIGN,
    DYG_SIGN,
    DYG_SIGN_NEW,
    OURS_LINKSIGN,
    OURS_NEW,
    E2S_FULL_PAT,
    load_seed_dir,
    pstd,
    series,
)

SIGN_KEYS = ["auc", "ap", "f1_binary", "f1_macro", "f1_weighted", "acc"]
LINKSIGN_KEYS = ["auc", "ap", "f1_mac", "f1_wt", "sign_f1", "exist_f1", "f1_mic", "acc"]


def dyg_sign_dir(ds: str) -> pathlib.Path:
    tpl = DYG_SIGN_NEW if ds in ("RedditHyperlinkTitle", "RedditHyperlinkBody") else DYG_SIGN
    return pathlib.Path(str(tpl).format(ds=ds))


def block(L, title, dir_fn, keys, pat="*.P1.TE.json"):
    L.append(f"## {title}\n")
    for ds in DATASETS:
        sm = load_seed_dir(dir_fn(ds), pat)
        if not sm:
            L.append(f"### {ds}\n\n⚠️ 缺档\n")
            continue
        seeds = sorted(sm)
        L.append(f"### {ds}\n")
        L.append("| 指标 | mean±pstd | " + " | ".join(f"seed{s}" for s in seeds) + " |")
        L.append("|---|---|" + "---|" * len(seeds))
        for k in keys:
            sv = series(sm, k)
            if not sv:
                continue
            vals = [sv[s] for s in seeds if s in sv]
            row = (
                f"| {k} | {sum(vals)/len(vals):.4f}±{pstd(vals):.4f} | "
                + " | ".join(f"{sv[s]:.4f}" for s in seeds if s in sv)
                + " |"
            )
            L.append(row)
        L.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/TABLES_valthr_20260917.md")
    args = ap.parse_args()

    L = []
    L.append("# val-thr 全指标表（2026-09-17）\n")
    L.append(
        "来源：真 DyGFormer #121–133（**val-thr 口径**；sign RT/RB 为刷新版）；"
        "ours sign #129–133（**full 口径**，val-thr）；ours linksign full（#75–82 批次）。"
    )
    L.append(
        "口径：`mean±pstd`（pstd=总体标准差 ddof=0）；JSON 原值为字符串；"
        "**auc/ap 对阈值不变**（val-thr 与旧 0.5-thr 逐位相同），f1_* / acc 为验证集选阈值结果。\n"
    )
    L.append("---\n")
    block(L, "A1. 真 DyGFormer · sign（2 类）", dyg_sign_dir, SIGN_KEYS)
    L.append("---\n")
    block(L, "A2. 真 DyGFormer · linksign（3 类）", lambda ds: pathlib.Path(str(DYG_LINKSIGN).format(ds=ds)), LINKSIGN_KEYS)
    L.append("---\n")
    block(L, "B. ours（SignDyGFormer，full 口径）· sign（2 类；val-thr）", lambda ds: pathlib.Path(str(OURS_NEW).format(ds=ds)), SIGN_KEYS)
    L.append("---\n")
    block(L, "C. ours（SignDyGFormer，full 口径）· linksign（3 类）", lambda ds: pathlib.Path(str(OURS_LINKSIGN).format(ds=ds)), LINKSIGN_KEYS, E2S_FULL_PAT)
    L.append("---\n")
    L.append("生成脚本：`tools/verify/export_valthr_tables.py`（可复现）。\n")

    out = pathlib.Path(args.out)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"[ok] 已写入 {out}（{len(L)} 行）")


if __name__ == "__main__":
    main()
