# -*- coding: utf-8 -*-
"""D6 附录底表：10 个 (任务×数据集) 配置 + 来源批次 + sha256（Paper 78f6 §四.2）。

口径（Paper 采纳）：定稿值 = 泄漏修复（09-11/12）后 5 种子复核批；早期候选来源不可回溯为泄漏时代专属。
输出：results/d6_config_source_table_20260925.{csv,md}
用法：python tools/verify/d6_config_table.py
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ["WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha", "BitcoinOTC"]
NL = {"WikiVote": (15, 10), "RedditHyperlinkTitle": (60, 1), "RedditHyperlinkBody": (80, 3),
      "BitcoinAlpha": (40, 15), "BitcoinOTC": (80, 5)}
NS = {"WikiVote": (40, 15), "RedditHyperlinkTitle": (100, 1), "RedditHyperlinkBody": (60, 1),
      "BitcoinAlpha": (40, 15), "BitcoinOTC": (40, 15)}
TAIL = {"WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody"}

NOTES = {
    ("sign", "BitcoinOTC"): "09-14 用户拍板换装 60/10→40/15（0.8775±0.0054 vs 0.8696±0.0048，配对 Δ+0.0079、t=3.16）；09-17 同步进 TASK 表",
    ("sign", "RedditHyperlinkTitle"): "ⓓ 候选 100/3 5 种子 n.s.（09-14）→ 未换装；定稿 = 修复后复核批",
    ("sign", "RedditHyperlinkBody"): "ⓓ 候选 40/1 5 种子 n.s.（09-14）→ 未换装；定稿 = 修复后复核批",
    ("sign", "WikiVote"): "修复后 5 种子复核批定稿（nh5/main）",
    ("sign", "BitcoinAlpha"): "修复后 5 种子复核批定稿（main-d）",
    ("linksign", "WikiVote"): "修复后主表批定稿（main-c）；E1c 探索列 m=8–11（未进主表）",
    ("linksign", "RedditHyperlinkTitle"): "修复后主表批定稿（main-b）",
    ("linksign", "RedditHyperlinkBody"): "修复后主表批定稿（main-b）；E1c 探索列 m=80（未进主表）",
    ("linksign", "BitcoinAlpha"): "修复后主表批定稿（main-e）",
    ("linksign", "BitcoinOTC"): "修复后主表批定稿（main-e）",
}

rows = []
for task, table, pat in (
    ("linksign", NL, "results/e1a_tailfill/raw_base/linksign/{ds}/*RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"),
    ("sign", NS, "results/sign_valthr/raw/{ds}/SignDyGFormer_seed42*.json"),
):
    for ds in DS:
        nn, lf = table[ds]
        files = [p for p in glob.glob(str(ROOT / pat.format(ds=ds))) if "-profiler" not in p]
        seed42 = [p for p in files if "seed42" in Path(p).name]
        assert len(seed42) == 1, f"{task}/{ds}: seed42 文件 {len(seed42)}"
        p = seed42[0]
        digest = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        rows.append({
            "task": task, "dataset": ds, "NN": nn, "LF": lf, "m": 0, "P": 1,
            "tail": "20000" if ds in TAIL else "—",
            "source_file": str(Path(p).relative_to(ROOT)).replace("\\", "/"),
            "sha256": digest,
            "note": NOTES[(task, ds)],
        })

csv_path = ROOT / "results/d6_config_source_table_20260925.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

L = []
ap_ = L.append
ap_("D6 附录底表：配置与来源（定稿值 = 泄漏修复后 5 种子复核批；早期候选来源不可回溯为泄漏时代专属）")
ap_("")
ap_("| 任务 | 数据集 | NN | LF | m | P | tail | 来源文件（seed42） | sha256[:16] | 备注 |")
ap_("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    ap_(f"| {r['task']} | {r['dataset']} | {r['NN']} | {r['LF']} | {r['m']} | {r['P']} | {r['tail']} | "
        f"`{r['source_file']}` | `{r['sha256'][:16]}` | {r['note']} |")
ap_("")
ap_("注：全部 10 行与 `run_experiments.py::TASK_DATASET_BEST_PARAMS` 一致；P=1 为默认（E-3 扫描无一致趋势）；m=0 主表默认。")
md_path = ROOT / "results/d6_config_source_table_20260925.md"
md_path.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"写出 {csv_path}")
print(f"写出 {md_path}")
print("\n".join(L))
