"""coverage_inventory.py — `6627` 实验覆盖盘点 + 成本估算（Code 2026-10-03）。

输出：results/coverage_inventory_20261003.txt
  A. 网格补 Bitcoin 成本（用 grid_new 实测单 run 时间推算）
  B. TE 补 Reddit 成本（用 sign_valthr RT/RB 实测单 run 时间）
  C. sign 侧消融库存（按 tag 组合统计各归档）
  D. 隔离/成对变体库存（linksign 侧）
  E. 指标可用键盘点（三键齐备性）
"""
from __future__ import annotations

import glob
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
L: list[str] = []
w = L.append


def runtimes(pat: str) -> list[float]:
    out = []
    for p in glob.glob(str(ROOT / pat)):
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
            v = d.get("single run time (s)") or d.get("training time (s)")
            out.append(float(v))
        except Exception:  # noqa: BLE001
            pass
    return out


def keys_of(pat: str) -> set[str]:
    for p in glob.glob(str(ROOT / pat)):
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        m = d.get("test metrics") or d.get("metrics") or {}
        return set(m.keys())
    return set()


w("`6627` 实验覆盖盘点 + 成本估算（Code 2026-10-03）")
w("=" * 100)
w("[A] 网格补 Bitcoin —— 成本推算（口径：2 ds × 2 task × 25 格 × 1 种子 = 100 runs）")
w("-" * 100)
for task in ("linksign", "sign"):
    for ds in ("RedditHyperlinkTitle", "RedditHyperlinkBody", "WikiVote"):
        ts = runtimes(f"results/grid_new/raw/{task}/{ds}/*.json")
        if ts:
            w(f"  grid_new {task:<9}{ds:<22} n={len(ts):<3} 单 run 均值 {st.mean(ts):7.1f}s  中位 {st.median(ts):7.1f}s")
w("  参考：Bitcoin 两集规模（BA 24,186 边 / OTC 35,592 边）与 RT/RB/WV 同量级；")
w("       按 RT/RB 实测均值 × 100 runs 估（下方合计）。")
w("")
w("[B] TE 补 Reddit —— 成本推算（口径：RT/RB × 3 策略 × 5 种子 = 30 runs，sign）")
w("-" * 100)
te_times = {}
for ds in ("RedditHyperlinkTitle", "RedditHyperlinkBody"):
    ts = runtimes(f"results/sign_valthr/raw/{ds}/*.json")
    te_times[ds] = st.mean(ts) if ts else None
    if ts:
        w(f"  sign_valthr {ds:<22} n={len(ts):<3} 单 run 均值 {st.mean(ts):7.1f}s")
w("")
w("[C] sign 侧消融库存（按 tag 组合统计；口径 = 同配置下存在的逐 run JSON 数）")
w("-" * 100)
ARCH = {
    "sign_valthr（full, 部署点）": "results/sign_valthr/raw/{ds}/*.json",
    "sign_wocnas（w/o CNAS）": "results/sign_wocnas/raw/{ds}/*.json",
    "sign_nobte（w/o BTE?）": "results/sign_nobte/raw/{ds}/*.json",
    "sign_rt5（tail-fill 5 种子）": "results/sign_rt5/raw/*.json",
    "sign_tailfill5": "results/sign_tailfill5/raw/{ds}/*.json",
    "sign_neighborhood": "results/sign_neighborhood/raw/{ds}/*.json",
    "sign_param（参数批）": "results/sign_param/raw/{ds}/*.json",
    "E-2_ablation/raw_seeds（2×2+变体）": "results/E-2_ablation/raw_seeds/{ds}/*.json",
}
for name, pat in ARCH.items():
    tot = 0
    per: list[str] = []
    for ds in ("BitcoinAlpha", "BitcoinOTC", "WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody"):
        n = len(glob.glob(str(ROOT / pat.format(ds=ds))))
        if n:
            per.append(f"{ds[:14]}={n}")
            tot += n
    w(f"  {name:<34} 总 {tot:>4} 件   " + "  ".join(per))
w("")
w("[D] linksign 侧隔离/成对变体库存（E-2_ablation/raw_seeds，按组合）")
w("-" * 100)
COMBOS = ["RAS-E.RASE-E.BTE-E.CNAS-E", "RAS-E.RASE-E.BTE-D.CNAS-E", "RAS-E.RASE-E.BTE-E.CNAS-D",
          "RAS-D.RASE-D.BTE-D.CNAS-D", "RAS-D.RASE-D.BTE-E.CNAS-E", "RAS-D.RASE-E.BTE-E.CNAS-E",
          "RAS-E.RASE-D.BTE-E.CNAS-E", "RAS-D.RASE-D.BTE-D.CNAS-E", "RAS-D.RASE-D.BTE-E.CNAS-D"]
w(f"  {'组合':<32}" + "".join(f"{d[:12]:>14}" for d in
                                ("BitcoinAlpha", "BitcoinOTC", "RedditHyperlinkTitle", "RedditHyperlinkBody", "WikiVote")))
for combo in COMBOS:
    cells = []
    for ds in ("BitcoinAlpha", "BitcoinOTC", "RedditHyperlinkTitle", "RedditHyperlinkBody", "WikiVote"):
        n = len(glob.glob(str(ROOT / f"results/E-2_ablation/raw_seeds/{ds}/*LF-Best.{combo}.P1.TE.json")))
        cells.append(f"{n:>14}")
    w(f"  {combo:<32}" + "".join(cells))
w("")
w("[E] 指标键盘点（三键齐备性；sign 任务需 f1_macro/f1_binary/auc）")
w("-" * 100)
for name, pat in (("sign_valthr", "results/sign_valthr/raw/BitcoinAlpha/*.json"),
                  ("te5/TE", "results/te5/raw/BitcoinAlpha/*.TE.json"),
                  ("grid_new/sign", "results/grid_new/raw/sign/BitcoinAlpha/seed42*"),
                  ("E-2_ablation/raw_seeds", "results/E-2_ablation/raw_seeds/WikiVote/*.json")):
    ks = sorted(keys_of(pat))
    w(f"  {name:<24} 键 = {ks}")
out = ROOT / "results/coverage_inventory_20261003.txt"
out.write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
print(f"[ok] {out.relative_to(ROOT)}")
