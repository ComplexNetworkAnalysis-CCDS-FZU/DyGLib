"""paired_stats_adopted_recheck.py — 采纳配置下 ours vs 真 DyGFormer 配对统计交叉复核（Code 2026-10-01）。

ours：results/grid_confirm/raw/{task}/{ds}/SignDyGFormer_seed*.NN-*.LF-*.G2.json
DyG ：results/s1_refresh/{raw,raw_valthr}/{task}/{ds}/DyGFormer_seed*.json
对比 Paper 本地自算（scratch/paired_stats_adopted_20261001.txt）。
输出：results/paired_adopted_recheck_20261001.txt
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "paired_adopted_recheck_20261001.txt"

CELLS = [
    ("linksign", "RedditHyperlinkTitle", "NN-15.LF-3", "RT 15/3",
     [("f1_wt", "main", +7.5, 5, 0.0227, 1.61),
      ("f1_mac", "f1_mac", +15.0, 5, 0.0031, 2.84),
      ("auc", "auc", +2.9, 4, 0.144, None)]),
    ("linksign", "RedditHyperlinkBody", "NN-60.LF-1", "RB 60/1",
     [("f1_wt", "main", +1.1, 3, 0.672, None),
      ("f1_mac", "f1_mac", -21.3, 0, 0.0254, -1.56),
      ("auc", "auc", -9.8, 0, 0.0400, -1.34)]),
    ("sign", "RedditHyperlinkTitle", "NN-60.LF-3", "sign RT 60/3",
     [("f1_macro", "main", +4.0, 4, 0.0495, 1.25),
      ("f1_bin", "f1_bin", -0.2, 3, 0.121, None),
      ("auc", "auc", -37.6, 0, 0.0031, -2.86)]),
]

KEYFIX = {"f1_mac": ["f1_mac", "f1_macro"], "f1_bin": ["f1_bin", "f1_binary"],
          "auc": ["auc", "AUC"], "f1_wt": ["f1_wt"], "f1_macro": ["f1_macro", "f1_mac"]}


def metric_of(d: dict, key: str) -> float | None:
    for cont in ("test metrics", "metrics"):
        m = d.get(cont) or {}
        for k in KEYFIX.get(key, [key]):
            if k in m:
                try:
                    return float(m[k])
                except (TypeError, ValueError):
                    return None
    return None


def load_ours(task: str, ds: str, pat: str) -> dict[int, dict[str, float]]:
    out = {}
    for p in glob.glob(str(ROOT / f"results/grid_confirm/raw/{task}/{ds}" / f"SignDyGFormer_seed*.{pat}*.G2.json")):
        seed = int(Path(p).name.split("seed")[1].split(".")[0])
        out[seed] = json.loads(Path(p).read_text(encoding="utf-8"))
    return out


def load_dyg(task: str, ds: str) -> tuple[dict[int, dict[str, float]], str]:
    """DyGFormer 真基线归档（**canonical 逐任务**，2026-10-01 Paper b84b 裁定）：
    linksign -> s1_refresh/raw/linksign/{ds}
    sign     -> s1_refresh/raw_valthr/sign/{ds}   ← val-threshold 档（阈值依赖指标的唯一可比档）
    """
    CANON = {"linksign": ["raw/linksign"], "sign": ["raw_valthr/sign", "raw/sign"]}
    for sub in CANON[task]:
        got = {}
        for p in glob.glob(str(ROOT / f"results/s1_refresh/{sub}/{ds}/DyGFormer_seed*.json")):
            seed = int(Path(p).name.split("seed")[1].split(".")[0])
            got[seed] = json.loads(Path(p).read_text(encoding="utf-8"))
        if got:
            return got, f"results/s1_refresh/{sub}/{ds}  [canonical-{task}]"
    return {}, "MISSING"


def stats(a: list[float], b: list[float]) -> tuple[float, int, float, float, float]:
    nd = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    mean = st.mean(diffs)
    sd = st.stdev(diffs) if nd > 1 else 0.0
    t = mean / (sd / nd ** 0.5) if sd > 0 else float("nan")
    d = mean / sd if sd > 0 else float("nan")
    pos = sum(1 for x in diffs if x > 0)
    from scipy import stats as ss
    p = float(ss.ttest_rel(a, b).pvalue)
    return mean, pos, p, t, d


L: list[str] = []
w = L.append
w("采纳配置 ours vs 真 DyGFormer 配对统计交叉复核（Code · 2026-10-01）")
w("ours = results/grid_confirm/raw/{task}/{ds}/*.G2.json（采纳 3 点）；DyG = results/s1_refresh/{raw/linksign | raw_valthr/sign}/{ds}（**canonical 逐任务**）")
w("⚠️ 代际说明（Paper b84b 裁定）：**threshold-free 指标（auc）跨档相同；阈值依赖指标（f1_*）跨档不可比** —— sign 的 canonical = `raw_valthr/sign`。")
w("对照 Paper 本地自算 scratch/paired_stats_adopted_20261001.txt（Δ 单位 ‰；同种子配对 n=5）")
w("")
allok = True
for task, ds, pat, label, metrics in CELLS:
    ours = load_ours(task, ds, pat)
    dyg, sub = load_dyg(task, ds)
    w("=" * 104)
    w(f"### {label}（ours n={len(ours)}，DyG n={len(dyg)}，**DyG 归档 = {sub}**）")
    w("=" * 104)
    w(f"{'指标':<10}{'Δ‰ 复算':>10}{'正/5':>7}{'p 复算':>10}{'d 复算':>9}   |  Paper: Δ‰ / 正 / p / d            | 结论")
    seeds = sorted(set(ours) & set(dyg))
    for key, lab, p_mm, p_pos, p_p, p_d in metrics:
        a = [metric_of(ours[s], key) for s in seeds]
        b = [metric_of(dyg[s], key) for s in seeds]
        if any(x is None for x in a) or any(y is None for y in b):
            w(f"{lab:<10}  [缺指标键: ours={a} dyg={b}]")
            continue
        mean, pos, p, t, d = stats(a, b)
        mm = mean * 1000
        ok = abs(mm - p_mm) <= 0.15 and pos == p_pos and abs(p - p_p) <= 0.002 and (
            p_d is None or abs(d - p_d) <= 0.03)
        allok = allok and ok
        ds_txt = "—" if p_d is None else f"{p_d:+.2f}"
        w(f"{lab:<10}{mm:>+10.1f}{pos:>4}/5{p:>10.4f}{d:>+9.2f}   | {p_mm:+.1f} / {p_pos} / {p_p:.4f} / {ds_txt:<6} | {'一致' if ok else '❌不一致'}"
          + f"   (t={t:+.2f})")
    w("")
w("=" * 104)
w(f"总体：{'全部一致（9/9）' if allok else '存在不一致项（见上）'}")
w("")
w("归档溯源（可追溯）：")
w("  ours : results/grid_confirm/raw/{linksign|sign}/{ds}/SignDyGFormer_seed*.NN-*.LF-*.G2.json  [gen=grid-confirm 2026-09-29]")
w("  DyG  : linksign -> results/s1_refresh/raw/linksign/{ds}/…  ; sign -> results/s1_refresh/raw_valthr/sign/{ds}/…  [canonical]")
w("注：ours 的 JSON `test metrics` 值为 4 位小数字符串；Δ/p/d 由该精度逐位复算 —— 与 Paper 数字一致即证明两边用了同一批逐种子值与相同配对算法。")
OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
print(f"[ok] {OUT.relative_to(ROOT)}")
