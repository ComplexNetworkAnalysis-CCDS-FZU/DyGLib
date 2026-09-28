# -*- coding: utf-8 -*-
"""从训练日志提取"最终 checkpoint 对应 epoch 的 val 指标"。

背景（2026-09-29 Paper a43d）：网格候选点确认批需先过 Gate 1（val 侧：候选点验证集
主指标 ≥ 当前点，同向不劣）。val 指标不在结果 JSON 中，须从训练日志解析。

日志格式（logs/SignDyGFormer/<ds>/SignDyGFormer_seed<seed>/<ts>.log）：
  每 epoch 打印 `... - root - INFO - validate <metric>, <value>`（"new node validate"
  另一块，须排除）；保存 checkpoint 时打印 `save model <path>.pkl`。
解析规则：跟踪当前 validate 块；每次遇 `save model ...pkl`（排除 hyper param）时快照
当前块；取最后一次快照 = 最终被测试的 checkpoint 的 val 指标。同时记录"save 时刻"。

用法（服务器仓库根；仅 stdlib）：
  python3 tools/verify/extract_val_from_logs.py                # 内置 Gate-1 十点（网格 seed42）
  python3 tools/verify/extract_val_from_logs.py --spec-file F  # 每行一条 spec: task,ds,nn,lf,seed[,suffix]
输出：TSV（制表符分隔），每 spec 一行。
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

# 内置 Gate-1 十点（网格屏幕批 seed42；cand=候选 点，ctrl=当前点）
DEFAULT_SPECS = [
    # task, ds, nn, lf, seed, note
    ("linksign", "RedditHyperlinkTitle", 15, 3, 42, "cand"),
    ("linksign", "RedditHyperlinkTitle", 60, 1, 42, "ctrl"),
    ("linksign", "RedditHyperlinkBody", 60, 1, 42, "cand"),
    ("linksign", "RedditHyperlinkBody", 80, 3, 42, "ctrl"),
    ("sign", "RedditHyperlinkTitle", 60, 3, 42, "cand"),
    ("sign", "RedditHyperlinkTitle", 100, 1, 42, "ctrl"),
    ("sign", "RedditHyperlinkBody", 40, 1, 42, "cand"),
    ("sign", "RedditHyperlinkBody", 60, 1, 42, "ctrl"),
    ("sign", "WikiVote", 15, 10, 42, "cand"),
    ("sign", "WikiVote", 40, 15, 42, "ctrl"),
]

TASK_DIR = {"linksign": "SignLinkPrediction", "sign": "LinkSign"}

VAL_RE = re.compile(r"- root - INFO - validate ([a-z_]+), ([-\d.eE]+)$")
SAVE_RE = re.compile(r"- root - INFO - save model (.+\.pkl)")

# 已观察到的 val 指标名（防把别的行误当指标）
KNOWN_METRICS = {
    "exist_recall", "exist_precision", "exist_f1", "sign_f1", "ap", "f1_mac",
    "f1_wt", "f1_mic", "acc", "auc", "auc_wt", "precision_neg", "recall_neg",
    "precision_pos", "recall_pos", "mcc", "thr_exist", "thr_sign",
}


def find_log(ds: str, seed: int, ckpt_base: str) -> str | None:
    """在 seed 目录下找含该 ckpt 名字的日志（多条取最新 mtime）。"""
    d = f"logs/SignDyGFormer/{ds}/SignDyGFormer_seed{seed}"
    cands = []
    for f in glob.glob(os.path.join(d, "*.log")):
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if "save model" in line and ckpt_base in line and line.rstrip().endswith(".pkl"):
                        cands.append(f)
                        break
        except OSError:
            continue
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def parse_log(path: str) -> dict:
    """返回最终 checkpoint 快照 + 保存时间 + epoch 数。"""
    cur: dict = {}
    snap: dict = {}
    snap_time = ""
    n_blocks = 0
    last_time = ""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = VAL_RE.search(line)
            if m and m.group(1) in KNOWN_METRICS:
                cur[m.group(1)] = float(m.group(2))
                last_time = line[:19]
                continue
            m = SAVE_RE.search(line)
            if m:
                if cur and ("thr_sign" in cur or "f1_wt" in cur or "f1_mac" in cur):
                    snap = dict(cur)
                    snap_time = line[:19]
                    n_blocks += 1
    return {"snap": snap, "snap_time": snap_time, "n_saves": n_blocks, "last_time": last_time}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec-file", default="", help="每行: task,ds,nn,lf,seed[,suffix]")
    ap.add_argument("--suffix", default="", help="附加后缀（如 .G2；默认空）")
    args = ap.parse_args()

    specs = []
    if args.spec_file:
        with open(args.spec_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                task, ds, nn, lf, seed = parts[0], parts[1], int(parts[2]), int(parts[3]), int(parts[4])
                note = parts[5] if len(parts) > 5 else ""
                specs.append((task, ds, nn, lf, seed, note))
    else:
        specs = DEFAULT_SPECS

    cols = ["task", "ds", "nn", "lf", "seed", "note", "log_mtime", "save_time",
            "n_saves", "f1_wt", "f1_mac", "auc", "ap", "sign_f1", "exist_f1",
            "f1_mic", "thr_sign", "thr_exist", "log_path"]
    print("\t".join(cols))
    for (task, ds, nn, lf, seed, note) in specs:
        base = (f"SignDyGFormer_seed{seed}.NN-{nn}.LF-{lf}"
                f".RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE{args.suffix}")
        ckpt_base = base
        logf = find_log(ds, seed, ckpt_base)
        if logf is None:
            print("\t".join([task, ds, str(nn), str(lf), str(seed), note,
                             "LOG_NOT_FOUND", "", "", "", "", "", "", "", "", "", "", base]))
            continue
        r = parse_log(logf)
        s = r["snap"]
        g = lambda k: (f"{s[k]:.4f}" if k in s else "")  # noqa: E731
        print("\t".join([
            task, ds, str(nn), str(lf), str(seed), note,
            f"{os.path.getmtime(logf):.0f}", r["snap_time"], str(r["n_saves"]),
            g("f1_wt"), g("f1_mac"), g("auc"), g("ap"), g("sign_f1"), g("exist_f1"),
            g("f1_mic"), g("thr_sign"), g("thr_exist"), logf,
        ]))
    print("", file=sys.stderr)


if __name__ == "__main__":
    main()
