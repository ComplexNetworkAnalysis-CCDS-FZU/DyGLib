# -*- coding: utf-8 -*-
"""BTE 边际三指标面导出（Paper ebcf 轻活 3；linksign，G1 配置下）。

批次（均非 TF 代，5 种子配对 42/123/456/789/1024）：
  full  = results/main_tables/raw/linksign/<ds>/SignDyGFormer_seed<s>.NN-<nn>.LF-<lf>...BTE-E...json
  loo   = results/E-2_ablation/raw_seeds/<ds>/SignDyGFormer_seed<s>.NN-Best.LF-Best.RAS-E.RASE-E.BTE-D.CNAS-E.P1.TE.json
  g1    = results/g1_gate/raw/linksign/<ds>/SignDyGFormer_seed<s>.NN-<nn>.LF-<lf>...BTE-E...G1.json

输出：
  results/bte_margin_20260924.csv   （逐 (ds, seed) 三列原始值：f1_wt/f1_mac/ap + auc）
  results/bte_margin_20260924.txt   （逐数据集边际 + 配对 t/p + 预登记判据式结论）
用法：python tools/verify/bte_margin_table.py（仓库根）
"""
from __future__ import annotations

import glob
import json
import math
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SEEDS = [42, 123, 456, 789, 1024]
DS = ["WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha", "BitcoinOTC"]
NNLF = {"WikiVote": (15, 10), "RedditHyperlinkTitle": (60, 1), "RedditHyperlinkBody": (80, 3),
        "BitcoinAlpha": (40, 15), "BitcoinOTC": (80, 5)}
METRICS = ["f1_wt", "f1_mac", "ap", "auc"]


def load(p: str):
    j = json.load(open(p, encoding="utf-8"))
    return {k: float(v) for k, v in j["test metrics"].items()}


def pick(pattern: str):
    hits = glob.glob(pattern)
    if len(hits) != 1:
        sys.exit(f"[ERR] glob 命中 {len(hits)} 个：{pattern}")
    return hits[0]


def t_paired(a, b):
    """配对 t 检验（a-b），返回 (均值差, t, p 双尾近似)。n=5。"""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = statistics.mean(d)
    sd = statistics.stdev(d)
    if sd == 0:
        return m, float("inf") if m != 0 else 0.0, 0.0 if m != 0 else 1.0
    t = m / (sd / math.sqrt(n))
    # t 分布双尾 p（df=n-1）用 scipy；无 scipy 时用 t 分布 CDF 近似→直接尝试 scipy
    try:
        from scipy import stats

        p = 2 * stats.t.sf(abs(t), df=n - 1)
    except Exception:
        p = float("nan")
    return m, t, p


def main():
    rows = []
    data = {}  # data[ds][batch][seed][metric]
    for ds in DS:
        nn, lf = NNLF[ds]
        data[ds] = {"full": {}, "loo": {}, "g1": {}}
        for s in SEEDS:
            full_p = pick(str(ROOT / f"results/main_tables/raw/linksign/{ds}/SignDyGFormer_seed{s}.NN-{nn}.LF-{lf}.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"))
            loo_p = pick(str(ROOT / f"results/E-2_ablation/raw_seeds/{ds}/SignDyGFormer_seed{s}.NN-Best.LF-Best.RAS-E.RASE-E.BTE-D.CNAS-E.P1.TE.json"))
            g1_p = pick(str(ROOT / f"results/g1_gate/raw/linksign/{ds}/SignDyGFormer_seed{s}.NN-{nn}.LF-{lf}.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.G1.json"))
            mf, ml, mg = load(full_p), load(loo_p), load(g1_p)
            data[ds]["full"][s], data[ds]["loo"][s], data[ds]["g1"][s] = mf, ml, mg
            rows.append((ds, s, mf, ml, mg))

    out_csv = ROOT / "results/bte_margin_20260924.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("dataset,seed,metric,full,loo_w_o_BTE,g1,g1_minus_full,full_minus_loo,g1_minus_loo\n")
        for ds, s, mf, ml, mg in rows:
            for k in METRICS:
                f.write(f"{ds},{s},{k},{mf[k]:.6f},{ml[k]:.6f},{mg[k]:.6f},"
                        f"{mg[k]-mf[k]:.6f},{mf[k]-ml[k]:.6f},{mg[k]-ml[k]:.6f}\n")

    L = []
    ap_ = L.append
    ap_("BTE 边际导出（linksign · G1 配置；三指标面 + AUC 对照；2026-09-24）")
    ap_("批次：full=main_tables（主表 full，非 TF）｜loo=E-2_ablation/raw_seeds w/o BTE（[T,T,F,T]，非 TF）｜g1=g1_gate（.G1）")
    ap_("公式：Δ(BTE|G1) = (G1 − full) + (full − w/o BTE) = G1 − w/o BTE（同 seed 配对）")
    ap_("")
    for k in METRICS:
        ap_(f"== 指标面 {k} ==")
        ap_(f"{'dataset':<22}{'full':>9}{'w/oBTE':>9}{'G1':>9}{'G1−full‰':>11}{'full−w/oBTE‰':>14}{'G1−w/oBTE‰':>13}   {'t':>7}{'p':>9}{'正/5':>5}")
        agg1, agg2, agg3 = [], [], []
        pos_ds = 0
        for ds in DS:
            a = [data[ds]["g1"][s][k] for s in SEEDS]
            b = [data[ds]["full"][s][k] for s in SEEDS]
            c = [data[ds]["loo"][s][k] for s in SEEDS]
            d1 = [x - y for x, y in zip(a, b)]
            d2 = [x - y for x, y in zip(b, c)]
            d3 = [x - y for x, y in zip(a, c)]
            m1 = statistics.mean(d1) * 1000
            m2 = statistics.mean(d2) * 1000
            m3 = statistics.mean(d3) * 1000
            _, t3, p3 = t_paired(a, c)
            npos = sum(1 for x in d3 if x > 0)
            pos_ds += 1 if m3 > 0 else 0
            agg1 += d1
            agg2 += d2
            agg3 += d3
            ap_(f"{ds:<22}{statistics.mean(b):>9.4f}{statistics.mean(c):>9.4f}{statistics.mean(a):>9.4f}"
                f"{m1:>11.1f}{m2:>14.1f}{m3:>13.1f}   {t3:>7.2f}{p3:>9.4f}{npos:>5}")
        _, ta, pa = 0.0, 0.0, 0.0  # 占位（逐数据集 t/p 已在上表；汇总用下方 25 配对）
        g_all = [data[d]["g1"][s][k] for d in DS for s in SEEDS]
        l_all = [data[d]["loo"][s][k] for d in DS for s in SEEDS]
        f_all = [data[d]["full"][s][k] for d in DS for s in SEEDS]
        _, t25a, p25a = t_paired(g_all, f_all)
        _, t25b, p25b = t_paired(f_all, l_all)
        _, t25c, p25c = t_paired(g_all, l_all)
        npos25 = sum(1 for x, y in zip(g_all, l_all) if x - y > 0)
        ap_(f"{'--25 配对汇总--':<22}{statistics.mean(f_all):>9.4f}{statistics.mean(l_all):>9.4f}{statistics.mean(g_all):>9.4f}"
            f"{statistics.mean([x-y for x,y in zip(g_all,f_all)])*1000:>11.1f}"
            f"{statistics.mean([x-y for x,y in zip(f_all,l_all)])*1000:>14.1f}"
            f"{statistics.mean([x-y for x,y in zip(g_all,l_all)])*1000:>13.1f}   "
            f"t/p(ΔG1−w/oBTE)={t25c:.2f}/{p25c:.4f}  正 {npos25}/25；数据集正 {pos_ds}/5")
        ap_("")
    out_txt = ROOT / "results/bte_margin_20260924.txt"
    out_txt.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"写出 {out_csv}")
    print(f"写出 {out_txt}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
