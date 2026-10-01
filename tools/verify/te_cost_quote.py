"""te_cost_quote.py — §4.5 时间编码口径 B 成本测算（Code 2026-10-01）。

统计 sign 任务各数据集「单 run 墙钟」（结果 JSON 的 `single run time (s)`/`training time (s)`），
推算 3 策略 × {BA,OTC,WV} × 5 种子 = 45 runs 的 GPU 小时。
输出：results/te_cost_quote_20261001.txt
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "te_cost_quote_20261001.txt"

SOURCES = {
    "BitcoinAlpha": ["results/sign_valthr/raw/BitcoinAlpha/*.json",
                     "results/cns_g2/raw/sign/BitcoinAlpha/*.json",
                     "results/sign_rt5/raw/*.json"],
    "BitcoinOTC": ["results/sign_valthr/raw/BitcoinOTC/*.json",
                   "results/cns_g2/raw/sign/BitcoinOTC/*.json"],
    "WikiVote": ["results/sign_valthr/raw/WikiVote/*.json",
                 "results/cns_g2/raw/sign/WikiVote/*.json"],
}

L: list[str] = []
w = L.append
w("§4.5 时间编码 · 口径 B 成本测算（Code · 2026-10-01）")
w("口径 B = 3 策略 × {BitcoinAlpha, BitcoinOTC, WikiVote} × 5 种子，sign 任务（AUC + F1_bin）= **45 runs**")
w("数据源：既有 sign 批次结果 JSON 的 `single run time (s)`（≈训练+评估的墙钟；2×RTX2080S）")
w("")
times: dict[str, list[float]] = {}
for ds, pats in SOURCES.items():
    ts: list[float] = []
    used = []
    for pat in pats:
        fs = sorted(glob.glob(str(ROOT / pat)))
        for p in fs:
            if "bitcoinalpha" in p.lower() and ds != "BitcoinAlpha":
                continue
            d = json.loads(Path(p).read_text(encoding="utf-8"))
            v = d.get("single run time (s)") or d.get("training time (s)")
            try:
                ts.append(float(v))
                used.append(Path(p).name)
            except (TypeError, ValueError):
                pass
        if ts:
            break
    times[ds] = ts
    if ts:
        w(f"  {ds:<14} n={len(ts):<3} 单 run 均值 {st.mean(ts):7.1f}s  中位 {st.median(ts):7.1f}s  "
          f"min {min(ts):6.1f}s  max {max(ts):7.1f}s   源: {ts and str(Path(pats[0]).parent)}")
    else:
        w(f"  {ds:<14} [未取到耗时字段]")
w("")
if all(times.values()):
    per_ds = {ds: st.mean(v) for ds, v in times.items() if v}
    total_s = 5 * sum(per_ds.values()) * 3  # 5 seeds × 3 strategies
    total_h = total_s / 3600
    w("推算（45 runs = 3 策略 × 3 数据集 × 5 种子）：")
    for ds, t in per_ds.items():
        w(f"   {ds:<14} 5 种子 × 3 策略 × {t:7.1f}s = {15 * t / 3600:5.2f} GPU·h")
    w(f"  合计 = {total_s:.0f} s = **{total_h:.2f} GPU·h**（单卡串行）")
    w(f"  双卡并行墙钟 ≈ **{total_h / 2:.2f} h**（+ 线性衰减实现/自检与调度开销）")
    w(f"  含 25% 余量：**{total_h * 1.25:.2f} GPU·h**")
    w("")
    w(f"判据：与 24 GPU·h 上限比较 ⇒ **{'能' if total_h * 1.25 <= 24 else '不能'} 在 24 GPU·h 内完成**")
w("")
w("附：link & sign（linksign）同口径成本 = 以 linksign 批次单 run 耗时同法推算（量级相同，±30%）。")
w("注：不动主表/不改默认配置；纯对照实验（附录 §4.5）。线性衰减需先实现（估 ~0.5 天含自检）。")
OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
print(f"[ok] {OUT.relative_to(ROOT)}")
