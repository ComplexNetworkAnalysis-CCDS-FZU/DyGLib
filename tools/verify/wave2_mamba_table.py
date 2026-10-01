"""wave2_mamba_table.py — 波二 DyG-Mamba 单模型表（只列 AUC；ScaDyG 退化不入表）。

输入：results/wave2/raw/DyG-Mamba/{ds}_seed{s}.json
输出：results/wave2_mamba_table_20261001.txt
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "wave2" / "raw" / "DyG-Mamba"
SCA = ROOT / "results" / "wave2" / "raw" / "ScaDyG"
OUT = ROOT / "results" / "wave2_mamba_table_20261001.txt"

DS = ["BitcoinAlpha", "BitcoinOTC", "WikiVote"]
SEEDS = [42, 123, 456, 789, 1024]

L: list[str] = []
w = L.append

w("波二 外部基线适配：DyG-Mamba 单模型表（Code · 2026-10-01）")
w("口径：3 ds × 5 seed；每 ds 5 种子 mean±std；仅列 AUC（adapter 未落盘 AP/acc —— 'AP/acc not dumped by the adapter'）。")
w("用途：**附录「外部基线适配」表**，不进入主表七方法排名池。")
w("")
w(f"{'数据集':<14}{'AUC mean':>10}{'std':>9}{'min':>9}{'max':>9}   逐种子")
w("-" * 88)
summ = {}
for ds in DS:
    v = [json.loads((RAW / f"{ds}_seed{s}.json").read_text(encoding="utf-8"))["metrics"]["AUC"] for s in SEEDS]
    summ[ds] = (st.mean(v), st.stdev(v), min(v), max(v))
    w(f"{ds:<14}{st.mean(v):>10.4f}{st.stdev(v):>9.4f}{min(v):>9.4f}{max(v):>9.4f}   "
      + " ".join(f"{x:.4f}" for x in v))
w("")
w("=" * 88)
w("ScaDyG（同批 15 件）：**全退化，不入表**（AUC ≡ 0.5000、val_AUC ≡ 0.5000；F1_bin = 多数类常数）")
w("=" * 88)
sc = [json.loads((SCA / f"{ds}_seed{s}.json").read_text(encoding="utf-8"))["metrics"]["AUC"] for ds in DS for s in SEEDS]
w(f"  15 件 AUC 唯一值 = {sorted(set(sc))} ；全等于 0.5 = {set(sc) == {0.5}}")
w("  根因（Baseline 618ba85 定位）：转换器边/节点特征为全零占位 ⇒ 编码器输入恒 0 ⇒ 打分常数 ⇔ AUC 恰 0.5；")
w("  另含 DGL 边序错配 + 训练历史窗 idx 偏移。修复后重跑中（Paper 给至 10-05；逾期改定性引用）。")
w("")
w("=" * 88)
w("附：Val AUC（同 run，供交叉核对；非论文表项）")
w("=" * 88)
for ds in DS:
    vv = [json.loads((RAW / f"{ds}_seed{s}.json").read_text(encoding="utf-8")).get("val_AUC") for s in SEEDS]
    w(f"  {ds:<14}" + " ".join(f"{x:.4f}" for x in vv) + f"   mean={st.mean(vv):.4f}")
w("")
w("注：DyG-Mamba 运行后端 = pure(vendor)（selective_scan_ref/mamba_inner_ref + pure LayerNorm/RMSNorm；")
w("    数学等价但非逐位一致），JSON 内 `note` 已标注 `mamba 内核后端=pure`。")

OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
print(f"[ok] {OUT.relative_to(ROOT)}")
