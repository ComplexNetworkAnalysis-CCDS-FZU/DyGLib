# -*- coding: utf-8 -*-
"""BTE 信号体检（离线 · 纯 CPU · 轻量）。

目的：在**测试期真实边**上直接检验「平衡理论证据」与「真实符号」的关联——
不训练模型，回答的问题是：BTE 通道想提取的信号，在数据里到底存不存在？

做法（忠实复刻 BTE 的计数规则，`NeighborInteractEncoder.sign_effect_count`）：
  - 对测试边 (u, v, t)：取两侧在 t 之前的近 K 条历史（K=--hist-cap，默认 100）；
  - 共同邻居 w：sign 乘积 s(u,w)*s(w,v) 的**出现对计数**（pos/neg，含重复边计数）；
  - 加权证据 score_w = pos - neg；另算**去重版**：每个 w 一票（sign(sum_u)*sign(sum_v)）；
  - 统计：覆盖率、符号命中率（sign(score)==真实符号）、AUC、分桶 P(+1)、
    以及对照：全局多数类、直接历史预测器（u-v 上一次符号）。

说明与边界：
  - 这是**乐观上界**：用全历史（截近 K）求交集，模型侧还要再经过 CNAS 窗口 + NN 截断，
    可见证据只会更少。因此：本体检"无信号" ⇒ 模型侧必然无信号；
    "有信号" ⇒ 还要看采样/实现是否把信号吃掉。
  - 只用 sign != 0 的真实边；测试段 = 最后 --test-ratio 分位之后。
  - 全部读列裁剪（u/i/ts/sign），小样本抽样（--edges），电池机器可跑。

用法（仓库根目录）：
    python tools/verify/bte_signal_check.py                    # 5 数据集 × 1500 边
    python tools/verify/bte_signal_check.py --edges 500 --datasets WikiVote BitcoinOTC
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]

# 数据集 → (csv 相对路径, 说明)；tail20000 与主实验口径一致
DATASETS = {
    "BitcoinAlpha": "processed_data/BitcoinAlpha/ml_BitcoinAlpha.csv",
    "BitcoinOTC": "processed_data/BitcoinOTC/ml_BitcoinOTC.csv",
    "WikiVote": "processed_data/WikiVote/ml_WikiVote_tail20000.csv",
    "RedditHyperlinkTitle": "processed_data/RedditHyperlinkTitle/ml_RedditHyperlinkTitle_tail20000.csv",
    "RedditHyperlinkBody": "processed_data/RedditHyperlinkBody/ml_RedditHyperlinkBody_tail20000.csv",
}


def load_edges(path: pathlib.Path):
    """读列裁剪：u, i, ts, sign（sign!=0 的真实边）。"""
    df = pd.read_csv(
        path,
        usecols=lambda c: c in {"u", "i", "ts", "sign"},
    )
    df = df[df["sign"] != 0].reset_index(drop=True)
    return (
        df["u"].to_numpy(np.int64),
        df["i"].to_numpy(np.int64),
        df["ts"].to_numpy(np.float64),
        df["sign"].to_numpy(np.int8),
    )


def build_node_histories(u, i, ts, sign):
    """每个节点的 (t, nbr, s) 历史，按时间升序。返回 dict[node] -> (times, nbrs, signs)。"""
    hist = {}
    for a, b, t, s in zip(u, i, ts, sign):
        for node, nbr in ((a, b), (b, a)):
            rec = hist.get(node)
            if rec is None:
                hist[node] = rec = ([], [], [])
            rec[0].append(t)
            rec[1].append(nbr)
            rec[2].append(s)
    out = {}
    for node, (tt, nn, ss) in hist.items():
        order = np.argsort(tt, kind="stable")
        out[node] = (
            np.asarray(tt, np.float64)[order],
            np.asarray(nn, np.int64)[order],
            np.asarray(ss, np.int8)[order],
        )
    return out


def recent_before(hist, node, t, cap):
    """节点 node 在时刻 t 之前最近的 cap 条历史（不含 t）。"""
    rec = hist.get(node)
    if rec is None:
        return np.empty(0, np.int64), np.empty(0, np.int8)
    times, nbrs, signs = rec
    cut = int(np.searchsorted(times, t, side="left"))
    if cut <= 0:
        return np.empty(0, np.int64), np.empty(0, np.int8)
    lo = max(0, cut - cap)
    return nbrs[lo:cut], signs[lo:cut]


def evidence(u_nbr, u_sign, v_nbr, v_sign):
    """返回 (pos, neg, vote_score, n_cn)。

    pos/neg：出现对计数（乘积=+1 记正证据；=-1 记负证据），与模型 BTE 一致；
    vote_score：去重版——每个共同邻居 w 一票（sign(Σs_u)*sign(Σs_v)）。
    """
    cu: dict[int, list] = {}
    for w, s in zip(u_nbr, u_sign):
        cu.setdefault(int(w), [0, 0])[0 if s == 1 else 1] += 1
    cv: dict[int, list] = {}
    for w, s in zip(v_nbr, v_sign):
        cv.setdefault(int(w), [0, 0])[0 if s == 1 else 1] += 1

    pos = neg = vote = 0
    n_cn = 0
    for w, (u_pos, u_neg) in cu.items():
        rec_v = cv.get(w)
        if rec_v is None:
            continue
        n_cn += 1
        v_pos, v_neg = rec_v
        pos += u_pos * v_pos + u_neg * v_neg
        neg += u_pos * v_neg + u_neg * v_pos
        su = np.sign(u_pos - u_neg)
        sv = np.sign(v_pos - v_neg)
        vote += int(su * sv)
    return pos, neg, vote, n_cn


def auc_score(score, label):
    """Mann-Whitney AUC（label: +1/-1）。"""
    pos = score[label == 1]
    neg = score[label == -1]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # 处理并列（平均秩）
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
    r_pos = ranks[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def direct_history_acc(hist, u, v, ts, sign):
    """直接历史预测器：u-v 上一次交互的符号（两侧任一）；返回 (acc@covered, coverage)。"""
    ok = tot = 0
    for a, b, t, s in zip(u, v, ts, sign):
        last = 0
        for node, other in ((a, b), (b, a)):
            rec = hist.get(node)
            if rec is None:
                continue
            times, nbrs, signs = rec
            cut = int(np.searchsorted(times, t, side="left"))
            if cut <= 0:
                continue
            idx = np.where(nbrs[:cut] == other)[0]
            if len(idx) > 0:
                last = int(signs[:cut][idx[-1]])
                break
        if last != 0:
            tot += 1
            ok += int(last == s)
    return (ok / tot if tot else float("nan")), (tot / len(u) if len(u) else float("nan"))


def run_dataset(name, csv_path, edges, cap, test_ratio, seed):
    path = ROOT / csv_path
    if not path.is_file():
        print(f"[skip] {name}: {path} 不存在")
        return None
    u, i, ts, sign = load_edges(path)

    # 历史必须用【全图】构建（train+val+test 全在），再对测试段逐边查询；
    # 查询时按时间截断（t 之前），与模型取历史的语义一致。
    hist = build_node_histories(u, i, ts, sign)

    test_time = float(np.quantile(ts, 1 - test_ratio))
    m = ts > test_time
    u, i, ts, sign = u[m], i[m], ts[m], sign[m]

    rng = np.random.default_rng(seed)
    n = min(edges, len(u))
    sel = rng.choice(len(u), size=n, replace=False)
    sel.sort()
    u, i, ts, sign = u[sel], i[sel], ts[sel], sign[sel]

    rows = []
    for a, b, t, s in zip(u, i, ts, sign):
        un, us = recent_before(hist, a, t, cap)
        vn, vs = recent_before(hist, b, t, cap)
        if len(un) == 0 or len(vn) == 0:
            rows.append((0, 0, 0, 0, int(s)))
            continue
        pos, neg, vote, n_cn = evidence(un, us, vn, vs)
        rows.append((pos, neg, vote, n_cn, int(s)))
    arr = np.asarray(rows, np.float64)
    pos, neg, vote, n_cn, sgn = (
        arr[:, 0],
        arr[:, 1],
        arr[:, 2],
        arr[:, 3],
        arr[:, 4],
    )

    cov = float((n_cn > 0).mean())
    ev = n_cn > 0
    score = pos - neg
    tie = ev & (score == 0)

    acc_cov = float((np.sign(score[~tie]) == sgn[~tie]).mean()) if (~tie).any() else float("nan")
    acc_vote = float((np.sign(vote[ev]) == sgn[ev]).mean()) if ev.any() else float("nan")
    auc_w = auc_score(score[ev], sgn[ev])
    auc_v = auc_score(vote[ev], sgn[ev])
    base_rate = float((sgn == 1).mean())
    majority = max(base_rate, 1 - base_rate)

    # 平衡准确率（对类别不平衡不敏感）：score>0 判正，score<0 判负，score==0 均抛一半
    def bal_acc(sc, lb):
        pos_m = lb == 1
        neg_m = lb == -1
        if score is None or pos_m.sum() == 0 or neg_m.sum() == 0:
            return float("nan")
        tpr = float((sc[pos_m] > 0).mean()) + 0.5 * float((sc[pos_m] == 0).mean())
        tnr = float((sc[neg_m] < 0).mean()) + 0.5 * float((sc[neg_m] == 0).mean())
        return 0.5 * (tpr + tnr)

    bal_w = bal_acc(score[ev], sgn[ev])
    bal_v = bal_acc(vote[ev], sgn[ev])
    d_acc, d_cov = direct_history_acc(hist, u, i, ts, sign)

    print(f"\n== {name} ==  测试段 {len(u)}（抽样 {n}）｜正例 {base_rate:.3f}")
    print(f"  覆盖率（≥1 共同邻居证据）: {cov:.3f}")
    print(f"  加权证据(pos-neg): 命中率 {acc_cov:.3f}｜AUC {auc_w:.3f}｜平衡准确率 {bal_w:.3f}")
    print(f"  去重投票        : 命中率 {acc_vote:.3f}｜AUC {auc_v:.3f}｜平衡准确率 {bal_v:.3f}")
    print(f"  对照：多数类={majority:.3f}｜直接历史预测器 acc={d_acc:.3f}（覆盖 {d_cov:.3f}）")
    buckets = [(1, 1), (2, 4), (5, 19), (20, 99), (100, 10**9)]
    print("  证据强度分桶（total=pos+neg → P(+1), n）:")
    total = pos + neg
    for lo, hi in buckets:
        m2 = ev & (total >= lo) & (total <= hi)
        if m2.any():
            print(
                f"    [{lo:>3},{hi if hi < 10**9 else '∞':>3}]: "
                f"P(+1)={float((sgn[m2] == 1).mean()):.3f}  n={int(m2.sum())}"
            )
    no_direct = None  # 直接历史子群分桶留给后续版本；此处仅报全量对照
    return {
        "name": name,
        "n": n,
        "cov": cov,
        "acc_w": acc_cov,
        "bal_w": bal_w,
        "auc_w": auc_w,
        "bal_v": bal_v,
        "auc_v": auc_v,
        "majority": majority,
        "direct_acc": d_acc,
        "direct_cov": d_cov,
    }


def main():
    ap = argparse.ArgumentParser(description="BTE 信号体检（离线/轻量）")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--edges", type=int, default=1500, help="每数据集抽样边数")
    ap.add_argument("--hist-cap", type=int, default=100, help="每侧最多取的历史条数")
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("BTE 信号体检（乐观上界：全历史截近 K；模型侧还有 CNAS 窗口 + NN 截断）")
    summary = []
    for name in args.datasets:
        csv_path = DATASETS.get(name)
        if csv_path is None:
            print(f"[skip] 未知数据集 {name}")
            continue
        r = run_dataset(name, csv_path, args.edges, args.hist_cap, args.test_ratio, args.seed)
        if r:
            summary.append(r)

    print("\n===== 汇总 =====  （bal_w/bal_v=平衡准确率；auc=排序信号；majority=多数类）")
    for r in summary:
        verdict = "有信号" if (r["auc_w"] > 0.55) else ("弱" if r["auc_w"] > 0.52 else "无")
        print(
            f"  {r['name']:<22s} cov={r['cov']:.3f}  bal_w={r['bal_w']:.3f}  "
            f"bal_v={r['bal_v']:.3f}  auc_w={r['auc_w']:.3f}  auc_v={r['auc_v']:.3f}  "
            f"maj={r['majority']:.3f}  → {verdict}"
        )


if __name__ == "__main__":
    main()
