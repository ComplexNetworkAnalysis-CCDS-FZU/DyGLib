# -*- coding: utf-8 -*-
"""D3b/T15：BTE 逐样本分组拆解（linksign；全零 vs 非全零子集；full / noBTE / G1）。

数据源：
  - 逐样本转储：results/samples_dump/{ds}/*.npz（seed42；含 exist/sign 概率、标签、批段长、阈值）
  - BTE 稀疏掩码：results/bte_sparsity/linksign_{ds}.npz（full 配置重放；real=sign!=0）
输出：results/d3b_grouped_20260924.txt
口径：
  - 主表 = 各 run 自带阈值（own-thr，协议原生）；稳健表 = 统一用 full 阈值（common-thr）；
  - 对齐校验：dump 正例数 == 掩码 real 数，且 dump 符号标签 == 掩码符号（1=正）；
  - 指标 = 仓库 get_linksign_prediction_metrics（is_logits=False；分组时按子集单次汇总）。
用法：python tools/verify/d3b_grouped_table.py（仓库根；本地即可）
"""
from __future__ import annotations

import glob
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.metrics.linkSignPredict import get_linksign_prediction_metrics  # noqa: E402

DS = ["WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha", "BitcoinOTC"]
METRICS = ["f1_wt", "f1_mac", "sign_f1", "exist_f1", "ap", "auc"]


def pick(paths):
    hits = glob.glob(paths)
    if len(hits) != 1:
        sys.exit(f"[ERR] glob 命中 {len(hits)}：{paths}")
    return hits[0]


def load_dump(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def extract_scores(d):
    """展开 dump：返回 (pos_exist, neg_exist, exist_y, sign_p, sign_y, ethr, sthr)。"""
    ep, ey = d["exist_p"], d["exist_y"]
    ebl, sbl = d["exist_batch_lens"], d["sign_batch_lens"]
    pos_e, neg_e = [], []
    off = 0
    for L, B in zip(ebl, sbl):
        seg = ep[off: off + L]
        off += L
        assert L == 2 * B, f"exist 批段长 {L} != 2B({B})"
        pos_e.append(seg[:B])
        neg_e.append(seg[B: 2 * B])
        # 校验段内 y 模式 = [1]*B+[0]*B
        assert (ey[off - L: off - B] == 1).all() and (ey[off - B: off] == 0).all()
    pos_e, neg_e = np.concatenate(pos_e), np.concatenate(neg_e)
    return (
        pos_e, neg_e,
        d["sign_p"], d["sign_y"].astype(np.int64),
        float(d["exist_thr"][0]), float(d["sign_thr"][0]),
    )


def metrics_for(pos_e, neg_e, sign_p, sign_y, idx, ethr, sthr):
    """子集（idx = 正例下标）上的 3class 指标（单次汇总口径）。"""
    e_p = np.concatenate([pos_e[idx], neg_e[idx]])
    e_y = np.concatenate([np.ones(len(idx)), np.zeros(len(idx))])
    s_p = sign_p[idx]
    s_y = sign_y[idx]
    if len(idx) == 0:
        return None
    m = get_linksign_prediction_metrics(
        torch.tensor(e_p), torch.tensor(e_y),
        torch.tensor(s_p), torch.tensor(s_y),
        ethr, sthr, is_logits=False,
    )
    return {k: float(m[k]) for k in METRICS}


def table_row(name, m):
    if m is None:
        return f"{name:<12}" + " " * 54 + "(空)"
    return f"{name:<12}" + "".join(f"{m[k]:>9.4f}" for k in METRICS)


def main():
    L = []
    ap_ = L.append
    ap_("D3b/T15 分组拆解（linksign · seed42 · full/noBTE/G1 × 全样本/全零子集/非全零子集）")
    ap_("口径：own-thr = 各 run 自带验证阈值；common-thr = 统一用 full 侧阈值。指标=仓库函数（is_logits=False）。")
    ap_("")
    per_ds = {}
    for ds in DS:
        mask_z = np.load(ROOT / f"results/bte_sparsity/linksign_{ds}.npz")
        real = mask_z["sign"] != 0
        sign_real = mask_z["sign"][real]
        az = mask_z["all_zero"][real]
        ratio = mask_z["ratio"][real]

        ddir = ROOT / f"results/samples_dump/{ds}"
        full = load_dump(pick(str(ddir / "*BTE-E.CNAS-E.P1.TE.EVT.npz")))
        nob = load_dump(pick(str(ddir / "*BTE-D*EVT.npz")))
        g1 = load_dump(pick(str(ddir / "*.G1.EVT.npz")))

        cfg = {}
        for tag, d in (("full", full), ("noBTE", nob), ("G1", g1)):
            cfg[tag] = extract_scores(d)
            # 对齐校验
            assert len(cfg[tag][2]) == real.sum() == len(az), \
                f"{ds}/{tag}: dump {len(cfg[tag][2])} vs mask real {real.sum()}"
            assert (cfg[tag][3] == (sign_real > 0)).all(), f"{ds}/{tag}: 符号标签与掩码顺序不一致"

        ethr_full = cfg["full"][4]
        sthr_full = cfg["full"][5]
        per_ds[ds] = (cfg, az, ratio, ethr_full, sthr_full)

        ap_(f"===== {ds}  |  全零占比 {az.mean()*100:.1f}%（{az.sum()}/{len(az)}）  "
            f"阈值 full=({ethr_full:.2f},{sthr_full:.2f}) noBTE={tuple(round(x,2) for x in cfg['noBTE'][4:6])} "
            f"G1={tuple(round(x,2) for x in cfg['G1'][4:6])}")
        ap_(f"{'config':<12}" + "".join(f"{k:>9}" for k in METRICS))
        idx_all = np.arange(len(az))
        idx_z = np.where(az)[0]
        idx_nz = np.where(~az)[0]
        for tag in ("full", "noBTE", "G1"):
            pos_e, neg_e, s_p, s_y, ethr, sthr = cfg[tag]
            ap_(f"-- own-thr")
            for nm, idx in (("ALL", idx_all), ("ZERO", idx_z), ("NONZERO", idx_nz)):
                ap_(table_row(f"{tag}:{nm}", metrics_for(pos_e, neg_e, s_p, s_y, idx, ethr, sthr)))
        ap_("")

    # ---- 汇总：关键差值（own-thr 与 common-thr 两口径；三指标面）----
    for thr_tag, use_common in (("own-thr", False), ("common-thr", True)):
        ap_(f"===== 关键差值（{thr_tag}；‰；正 = 前者更大）=====")
        for key in ("f1_wt", "sign_f1", "auc"):
            ap_(f"-- {key}")
            ap_(f"{'ds':<22}{'Δ(noBTE−full) ZERO':>20}{'Δ(noBTE−full) NONZERO':>23}"
                f"{'Δ(G1−full) ZERO':>18}{'Δ(G1−full) NONZERO':>20}")
            for ds in DS:
                cfg, az, ratio, ethr_full, sthr_full = per_ds[ds]
                idx_z, idx_nz = np.where(az)[0], np.where(~az)[0]
                row = []
                for tag in ("full", "noBTE", "G1"):
                    pos_e, neg_e, s_p, s_y, ethr, sthr = cfg[tag]
                    if use_common:
                        ethr, sthr = ethr_full, sthr_full
                    m_z = metrics_for(pos_e, neg_e, s_p, s_y, idx_z, ethr, sthr)
                    m_nz = metrics_for(pos_e, neg_e, s_p, s_y, idx_nz, ethr, sthr)
                    row.append((m_z, m_nz))
                (fz, fnz), (bz, bnz), (gz, gnz) = row
                ap_(f"{ds:<22}"
                    f"{(bz[key]-fz[key])*1000:>20.1f}{(bnz[key]-fnz[key])*1000:>23.1f}"
                    f"{(gz[key]-fz[key])*1000:>18.1f}{(gnz[key]-fnz[key])*1000:>20.1f}")
            ap_("")

    out = ROOT / "results/d3b_grouped_20260924.txt"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"写出 {out}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
