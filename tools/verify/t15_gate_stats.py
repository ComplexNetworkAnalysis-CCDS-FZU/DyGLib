# -*- coding: utf-8 -*-
"""T15 附件：G1 门控值分布（由 bte_sparsity 掩码推导；位置级证据占比 + 逐样本 ratio 分位）。

G1 门 = 0/1（位置有证据 ⇒ 1）。位置级开概率 = Σ(ratio·L_eff)/Σ(L_eff)（real 边集合）。
输出：results/t15_gate_distribution_20260925.txt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DS = ["WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha", "BitcoinOTC"]

L = []
ap_ = L.append
ap_("T15 门控值分布（G1=0/1；掩码口径；linksign seed42 配置）")
ap_(f"{'ds':<22}{'位置开概率':>12}{'全零样本%':>10}{'ratio p25':>10}{'p50':>8}{'p75':>8}{'样本数':>8}")
for ds in DS:
    z = np.load(ROOT / f"results/bte_sparsity/linksign_{ds}.npz")
    real = z["sign"] != 0
    ratio = z["ratio"][real]
    le = z["L_eff"][real].astype(float)
    valid = np.isfinite(ratio) & (le > 0)
    frac = float((ratio[valid] * le[valid]).sum() / le[valid].sum())
    az = float(z["all_zero"][real].mean() * 100)
    q = np.nanquantile(ratio, [0.25, 0.5, 0.75])
    ap_(f"{ds:<22}{frac*100:>11.1f}%{az:>9.1f}%{q[0]:>10.3f}{q[1]:>8.3f}{q[2]:>8.3f}{int(real.sum()):>8}")
ap_("")
ap_("注：G1 门=0 仅作用于无证据位置；位置开概率即'有证据位置占比'（两侧序列合并计权）。")

out = ROOT / "results/t15_gate_distribution_20260925.txt"
out.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"写出 {out}")
print("\n".join(L))
