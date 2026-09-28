# -*- coding: utf-8 -*-
"""从训练日志提取"最终 checkpoint 对应 epoch 的 val 指标"。

背景（2026-09-29 Paper a43d）：网格候选点确认批需先过 Gate 1（val 侧：候选点验证集
主指标 ≥ 当前点，同向不劣）。val 指标不在结果 JSON 中，须从训练日志解析。

日志格式（logs/SignDyGFormer/<ds>/SignDyGFormer_seed<seed>/<ts>.log）：
  每 epoch 打印 `... - root - INFO - validate <metric>, <value>`（"new node validate"
  另一块，须排除）；保存 checkpoint 时打印 `save model <path>.pkl`。
解析规则：跟踪当前 validate 块；每次遇 `save model ...pkl`（排除 hyper param）时快照
当前块；取最后一次快照 = 最终被测试的 checkpoint 的 val 指标。同时记录"save 时刻"。

匹配注意（2026-09-29 教训）：同名 ckpt 历史多次出现（跨任务/跨批），必须同时校验
① 保存路径含 `./saved_models/{SignLinkPrediction|LinkSign}/`（任务过滤）；
② 文件名以 `<ckpt_base>.pkl` **精确结尾**（防 `.TF-E.RK-*` / `.E2-*` 后缀件误配）。

用法（服务器仓库根；仅 stdlib）：
  python3 tools/verify/extract_val_from_logs.py                     # 内置 Gate-1 十点
  python3 tools/verify/extract_val_from_logs.py --spec-file F       # 每行: task,ds,nn,lf,seed[,note]
  python3 tools/verify/extract_val_from_logs.py --spec-file F --window "09-25,09-27"
  --newest-only：每 spec 只输出最新匹配。
输出：每匹配日志一行 JSON（含全部 val 指标）。
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


def all_matches(task: str, ds: str, seed: int, ckpt_base: str) -> list:
    """找所有含该 ckpt 精确保存行的日志（任务路径过滤：SignLinkPrediction/LinkSign）。"""
    d = f"logs/SignDyGFormer/{ds}/SignDyGFormer_seed{seed}"
    tdir = TASK_DIR[task]
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.log"))):
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    m = SAVE_RE.search(line)
                    if m and f"/saved_models/{tdir}/" in line and m.group(1).endswith(f"/{ckpt_base}.pkl"):
                        out.append(f)
                        break
        except OSError:
            continue
    return out


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


def in_window(path: str, window: str) -> bool:
    if not window:
        return True
    lo, hi = [w.strip() for w in window.split(",")]
    import time
    t = time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(path)))
    return lo <= t <= hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec-file", default="", help="每行: task,ds,nn,lf,seed[,note]")
    ap.add_argument("--suffix", default="", help="名称后缀（如 .G2；默认空）")
    ap.add_argument("--window", default="", help='mtime 窗口过滤，如 "09-25 00:00,09-27 00:00"')
    ap.add_argument("--newest-only", action="store_true", help="每 spec 只输出最新一条匹配")
    args = ap.parse_args()

    if args.spec_file:
        specs = []
        with open(args.spec_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                note = parts[5] if len(parts) > 5 else ""
                specs.append((parts[0], parts[1], int(parts[2]), int(parts[3]), int(parts[4]), note))
    else:
        specs = DEFAULT_SPECS

    for (task, ds, nn, lf, seed, note) in specs:
        base = (f"SignDyGFormer_seed{seed}.NN-{nn}.LF-{lf}"
                f".RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE{args.suffix}")
        matches = [f for f in all_matches(task, ds, seed, base) if in_window(f, args.window)]
        matches = sorted(matches, key=os.path.getmtime)
        if args.newest_only:
            matches = matches[-1:]
        if not matches:
            print(json.dumps({"task": task, "ds": ds, "nn": nn, "lf": lf, "seed": seed,
                              "note": note, "base": base, "status": "NO_MATCH"}, ensure_ascii=False))
            continue
        for f in matches:
            r = parse_log(f)
            print(json.dumps({
                "task": task, "ds": ds, "nn": nn, "lf": lf, "seed": seed, "note": note,
                "log": f, "log_mtime": round(os.path.getmtime(f)),
                "n_saves": r["n_saves"], "save_time": r["snap_time"],
                "metrics": {k: round(v, 4) for k, v in sorted(r["snap"].items())},
            }, ensure_ascii=False))


if __name__ == "__main__":
    main()
