"""te5_table.py — §4.5 三策略（TE/cosine vs TD/exp vs TD-LIN/linear）对照表（Code 2026-10-03）。

输入：results/te5/raw/{ds}/SignDyGFormer_seed*.NN-40.LF-15…P1.{TE,TD,TD-LIN}.json
留档：results/te_probe_precheck_20261002.txt（主口径同名文件的 sha256 + 逐种子值）
输出：results/te5_table_20261003.txt

口径：sign 任务；指标 = f1_macro(主) / f1_binary / auc；mean±std 用 **ddof=1**；
     配对比较 = 同种子 vs TE 臂（n=5，配对 t）。
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "te5" / "raw"
PRE = ROOT / "results" / "sign_valthr" / "raw"
OUT = ROOT / "results" / "te5_table_20261003.txt"

DS = ["BitcoinAlpha", "BitcoinOTC", "WikiVote"]
SEEDS = [42, 123, 456, 789, 1024]
ARMS = ["TE", "TD", "TD-LIN"]
ARMN = {"TE": "cosine(TE)", "TD": "exp(TD)", "TD-LIN": "linear(TD-LIN)"}
KEYS = [("f1_macro", "F1_mac★"), ("f1_binary", "F1_bin"), ("auc", "AUC")]

L: list[str] = []
w = L.append


def load(ds: str, tag: str) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for s in SEEDS:
        p = RAW / ds / f"SignDyGFormer_seed{s}.NN-40.LF-15.RAS-E.RASE-E.BTE-E.CNAS-E.P1.{tag}.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        out[s] = d.get("test metrics", {})
    return out


w("§4.5 时间编码三策略对照表（sign 任务；同代际/同配置/同种子/同 run 同 ckpt 重跑）· Code 2026-10-03")
w("配置：各数据集 sign 部署点 NN-40 / LF-15（WV 另 --tail-num 20000）；λ=0.1、Δt 口径 staleness(A)；")
w("      TD-LIN 的 γ 按『与 exp 在 Δt 中位数处等权』自动标定（Paper d661 §二 批准）。")
w("口径：mean±std 用 ddof=1；配对 = 同种子 vs TE 臂（n=5）。E-4（09-12）旧代际数字**不复用**。")
w("")
w("=" * 104)
w("【核对】TE 臂（重跑）vs 主口径旧档（results/sign_valthr/raw；启动前已留 sha256）")
w("=" * 104)
same_all = True
for ds in DS:
    te = load(ds, "TE")
    for s in SEEDS:
        p = PRE / ds / f"SignDyGFormer_seed{s}.NN-40.LF-15.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
        ov = json.loads(p.read_text(encoding="utf-8")).get("test metrics", {})
        for k, _ in KEYS:
            a, b = te[s].get(k), ov.get(k)
            if a != b:
                same_all = False
                w(f"  ! {ds}/seed{s}/{k}: 重跑={a} 旧档={b}")
w(f"  结论：**TE 臂 ≡ 主口径（旧档）逐位一致 = {same_all}**（3 ds × 5 种子 × 3 指标）")
w("  ⇒ 三策略对照在同一口径下成立；旧档未被污染（若为 False 则须先查代际）。")
w("")
for ds in DS:
    arms = {t: load(ds, t) for t in ARMS}
    w("=" * 104)
    w(f"### {ds}")
    w("=" * 104)
    for k, kn in KEYS:
        w(f"--- {kn}（mean±std, ddof=1；Δ‰ 与 p 为同种子配对 vs cosine）")
        base = [float(arms["TE"][s][k]) for s in SEEDS]
        w(f"{'策略':<16}{'mean':>9}{'std':>9}{'Δ‰':>9}{'正/5':>6}{'p':>9}    逐种子")
        for t in ARMS:
            v = [float(arms[t][s][k]) for s in SEEDS]
            if t == "TE":
                w(f"{ARMN[t]:<16}{st.mean(v):>9.4f}{st.stdev(v):>9.4f}{'—':>9}{'—':>6}{'—':>9}    "
                  + " ".join(f"{x:.4f}" for x in v))
                continue
            diffs = [a - b for a, b in zip(v, base)]
            from scipy import stats as ss
            p_ = float(ss.ttest_rel(v, base).pvalue)
            pos = sum(1 for x in diffs if x > 0)
            w(f"{ARMN[t]:<16}{st.mean(v):>9.4f}{st.stdev(v):>9.4f}{st.mean(diffs) * 1000:>+9.1f}"
              f"{pos:>4}/5{p_:>9.4f}    " + " ".join(f"{x:.4f}" for x in v))
        w("")
w("=" * 104)
w("【一句话判读（供 §4.5 正文）】")
w("=" * 104)
for ds in DS:
    arms = {t: load(ds, t) for t in ARMS}
    v = {t: [float(arms[t][s]["f1_macro"]) for s in SEEDS] for t in ARMS}
    a = {t: [float(arms[t][s]["auc"]) for s in SEEDS] for t in ARMS}
    w(f"  {ds:<14} F1_mac: cosine {st.mean(v['TE']):.4f} | exp {st.mean(v['TD']):.4f} "
      f"({(st.mean(v['TD']) - st.mean(v['TE'])) * 1000:+.1f}‰) | linear {st.mean(v['TD-LIN']):.4f} "
      f"({(st.mean(v['TD-LIN']) - st.mean(v['TE'])) * 1000:+.1f}‰)"
      f"   ‖ AUC: cosine {st.mean(a['TE']):.4f} | exp {st.mean(a['TD']):.4f} | linear {st.mean(a['TD-LIN']):.4f}")
OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
print(f"[ok] {OUT.relative_to(ROOT)}")
