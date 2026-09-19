# -*- coding: utf-8 -*-
"""全指标绝对值表（Paper 补数 2026-09-19）+ sign 预测分布退化检查。

用法（仓库根）：
    python tools/verify/metrics_abs_table.py

内容：
A) linksign（results/E-2_ablation/raw_seeds/{ds}）full + 4 掩码 × 全键 mean±pstd
   （键：auc(OVR-macro 旧口径) / ap / sign_f1 / exist_f1 / f1_mic / f1_mac / f1_wt / acc）
B) sign：full（results/sign_valthr/raw/{ds}，CNAS-E）vs w/o CNAS（results/sign_wocnas/raw/{ds}）
   × 全键 mean±pstd（auc/ap/f1_binary/f1_macro/f1_weighted/acc；
   w/o CNAS 侧另有 precision/recall/balanced_acc/mcc/thr）
C) sign 逐种子列举 + 退化形态标记（balanced_acc ≤0.52 或 mcc ≤0.02 → ⚠）；
   负类召回由 2*balanced_acc − recall 反推（标 *）。

注：E-2 掩码与 sign_valthr 批为扩展指标契约（2026-09-18）前的旧代码产物，
无 auc_wt/precision 等新键；w/o CNAS（sign_wocnas 批）为新代码产物。
"""
import glob
import json
import math
import pathlib
import re

SEED_RE = re.compile(r"seed(\d+)")
DATASETS = ["RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha", "BitcoinOTC", "WikiVote"]
FULL = "RAS-E.RASE-E.BTE-E.CNAS-E"
CONFIGS = [
    ("full", FULL),
    ("w/o CNAS", "RAS-E.RASE-E.BTE-E.CNAS-D"),
    ("w/o BTE", "RAS-E.RASE-E.BTE-D.CNAS-E"),
    ("w/o RAE", "RAS-E.RASE-D.BTE-E.CNAS-E"),
    ("w/o RAS", "RAS-D.RASE-E.BTE-E.CNAS-E"),
]
LINK_KEYS = ["auc", "ap", "sign_f1", "exist_f1", "f1_mic", "f1_mac", "f1_wt", "acc"]
SIGN_KEYS = ["auc", "ap", "f1_binary", "f1_macro", "f1_weighted", "acc"]
SIGN_NEW = ["precision", "recall", "balanced_acc", "mcc", "thr"]


def pstd(vals):
    if len(vals) < 2:
        return 0.0
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))


def agg(rec, keys):
    cells = []
    for k in keys:
        vals = [float(v[k]) for v in rec.values() if k in v]
        if not vals:
            cells.append("-".ljust(19))
        else:
            mu = sum(vals) / len(vals)
            cells.append(f"{mu:.4f}±{pstd(vals):.4f}".ljust(19))
    return cells


def load_linksign(ds, mask):
    pat = (
        "results/E-2_ablation/raw_seeds/"
        + ds
        + "/SignDyGFormer_seed*.NN-Best.LF-Best."
        + mask
        + ".P1.TE.json"
    )
    out = {}
    for f in glob.glob(pat):
        sm = SEED_RE.search(pathlib.Path(f).name)
        if sm:
            out[int(sm.group(1))] = json.load(open(f, encoding="utf-8")).get("test metrics", {})
    return out


def load_sign(base, ds, cnas_flag):
    out = {}
    for f in glob.glob("results/" + base + "/raw/" + ds + "/*.json"):
        name = pathlib.Path(f).name
        if cnas_flag not in name:
            continue
        sm = SEED_RE.search(name)
        if sm:
            out[int(sm.group(1))] = json.load(open(f, encoding="utf-8")).get("test metrics", {})
    return out


def section_a():
    print("=" * 110)
    print("A) linksign 全键绝对值（5 种子 mean±pstd；auc = OVR-macro 旧口径，本批无 auc_wt）")
    print("=" * 110)
    for ds in DATASETS:
        print(f"\n-- {ds} --")
        header = "config".ljust(10) + "n |" + "|".join(k.center(19) for k in LINK_KEYS)
        print(header)
        for label, mask in CONFIGS:
            rec = load_linksign(ds, mask)
            cells = agg(rec, LINK_KEYS)
            print(f"{label:<10}{len(rec):>2} |" + "|".join(cells))


def section_b():
    print("\n" + "=" * 110)
    print("B) sign 全键绝对值（5 种子 mean±pstd）full vs w/o CNAS")
    print("=" * 110)
    for ds in DATASETS:
        print(f"\n-- {ds} --")
        header = "config".ljust(10) + "n |" + "|".join(k.center(19) for k in SIGN_KEYS)
        print(header)
        full = load_sign("sign_valthr", ds, "CNAS-E")
        woc = load_sign("sign_wocnas", ds, "CNAS-D")
        print(f"{'full':<10}{len(full):>2} |" + "|".join(agg(full, SIGN_KEYS)))
        print(f"{'w/o CNAS':<10}{len(woc):>2} |" + "|".join(agg(woc, SIGN_KEYS)))
        cells2 = agg(woc, SIGN_NEW)
        print("  w/o CNAS 扩展键：" + " | ".join(f"{k}={c.strip()}" for k, c in zip(SIGN_NEW, cells2)))


def section_c():
    print("\n" + "=" * 110)
    print("C) sign 逐种子 + 退化标记（[DEGEN] = balanced_acc<=0.52 或 mcc<=0.02；*负类召回=2*bal_acc-recall 反推）")
    print("=" * 110)
    for ds in DATASETS:
        print(f"\n-- {ds} --")
        full = load_sign("sign_valthr", ds, "CNAS-E")
        woc = load_sign("sign_wocnas", ds, "CNAS-D")
        print("  full（旧代码：无 bal_acc/mcc）")
        for s in sorted(full):
            m = full[s]
            print(
                f"    seed{s:<5} auc={float(m['auc']):.4f} f1_bin={float(m['f1_binary']):.4f} "
                f"f1_mac={float(m['f1_macro']):.4f} acc={float(m['acc']):.4f}"
            )
        print("  w/o CNAS（新代码：含 bal_acc/mcc/thr/逐类 P/R）")
        for s in sorted(woc):
            m = woc[s]
            bal = float(m.get("balanced_acc", "nan"))
            mcc = float(m.get("mcc", "nan"))
            rec = float(m.get("recall", "nan"))
            rec_neg = 2 * bal - rec if bal == bal and rec == rec else float("nan")
            flag = " [DEGEN]" if (bal == bal and bal <= 0.52) or (mcc == mcc and mcc <= 0.02) else ""
            print(
                f"    seed{s:<5} auc={float(m['auc']):.4f} f1_bin={float(m['f1_binary']):.4f} "
                f"f1_mac={float(m['f1_macro']):.4f} acc={float(m['acc']):.4f} "
                f"bal_acc={bal:.4f} mcc={mcc:+.4f} prec={float(m.get('precision', float('nan'))):.4f} "
                f"rec={rec:.4f} rec_neg*={rec_neg:.4f} thr={float(m.get('thr', float('nan'))):.4f}{flag}"
            )


def main():
    section_a()
    section_b()
    section_c()


if __name__ == "__main__":
    main()
