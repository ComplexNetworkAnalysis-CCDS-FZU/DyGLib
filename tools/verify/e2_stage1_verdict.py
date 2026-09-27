"""E2 段一方向判定（门禁②，2026-09-27；seed42 单种子方向屏）。

预登记口径（Paper af3e §四）：
  - 靶心：RedditBody(linksign) 的 f1_wt 转正；
  - 一般判据：Δ≥0.005 + 配对显著(p<.05) + ≥3/5 数据集同向（显著性属段二 5 种子范畴）；
  - 段一 → 报方向 + 段二胜者 k 建议；双口径（FX）待段二。

输入：
  E2:   results/e2_stage1/raw/{linksign|sign}/{ds}/*.E2-{3,10,30}.json （seed42）
  Full: linksign → results/e1a_tailfill/raw_base/linksign/{ds}/SignDyGFormer_seed42.*.json（同代际）
        sign     → results/sign_valthr/raw/{ds}/SignDyGFormer_seed42.*.json
输出：results/e2_stage1_verdict_20260927.txt（幂等覆盖）
用法：python tools/verify/e2_stage1_verdict.py
"""
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/e2_stage1_verdict_20260927.txt"

DS = ["WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha", "BitcoinOTC"]
KS = [3, 10, 30]
TASKS = [
    # (task, e2 目录, full 目录模板, 主指标, 次指标)
    ("linksign", "results/e2_stage1/raw/linksign", "results/e1a_tailfill/raw_base/linksign", ["f1_wt", "f1_mac", "auc"]),
    ("sign", "results/e2_stage1/raw/sign", "results/sign_valthr/raw", ["f1_binary", "f1_macro", "auc"]),
]

lines = []


def say(s=""):
    lines.append(s)
    print(s)


def load_json(p):
    return json.load(open(p, encoding="utf-8"))["test metrics"]


def find_e2(task_dir, ds, k):
    hits = glob.glob(str(ROOT / task_dir / ds / f"*.E2-{k}.json"))
    assert len(hits) == 1, f"E2 件不唯一/缺失: {task_dir}/{ds} k={k}: {hits}"
    return load_json(hits[0])


def find_full(full_dir, ds):
    hits = [p for p in glob.glob(str(ROOT / full_dir / ds / "SignDyGFormer_seed42.*.json")) if "E2-" not in p]
    assert len(hits) == 1, f"Full 件不唯一/缺失: {full_dir}/{ds}: {hits}"
    return load_json(hits[0])


say("E2 段一方向判定（门禁②；seed42 单种子方向屏；Δ‰ 相对同代际 Full）")
say("口径（af3e 预登记）：靶心=RB(linksign) f1_wt 转正；一般=Δ≥+5‰ 且 ≥3/5 同向（显著性/双口径 FX 属段二）")
say()

summary = {}  # (task, k) -> dict(primary deltas)
for task, e2_dir, full_dir, metrics in TASKS:
    say(f"===== {task}（主指标 {metrics[0]}；Δ‰ = E2−Full）=====")
    hdr = f"{'ds':<22s}"
    for k in KS:
        hdr += f"{'k=' + str(k):>26s}"
    say(hdr)
    sub = f"{'':22s}"
    for k in KS:
        sub += "".join(f"{m:>9s}" for m in metrics)
    say(sub)
    for ds in DS:
        row = f"{ds:<22s}"
        for k in KS:
            e2 = find_e2(e2_dir, ds, k)
            full = find_full(full_dir, ds)
            for m in metrics:
                d = (float(e2[m]) - float(full[m])) * 1000
                summary.setdefault((task, k), {}).setdefault(ds, {})[m] = d
                row += f"{d:>+9.1f}"
        say(row)
    say("  按 k 汇总（主指标）：")
    for k in KS:
        vals = [summary[(task, k)][ds][metrics[0]] for ds in DS]
        pos = sum(1 for v in vals if v > 0)
        ge5 = sum(1 for v in vals if v >= 5)
        mean = sum(vals) / len(vals)
        detail = " ".join(f"{ds[:4]}={v:+.1f}" for ds, v in zip(DS, vals))
        say(f"    k={k:<2d}  正={pos}/5  ≥+5‰={ge5}/5  均值={mean:+.1f}‰  [{detail}]")
    say()

# 靶心
say("===== 靶心（RB linksign f1_wt）=====")
for k in KS:
    v = summary[("linksign", k)]["RedditHyperlinkBody"]["f1_wt"]
    say(f"  k={k:<2d}  Δ = {v:+.1f}‰ {'✅ 转正' if v > 0 else '❌ 未转正'}")
say()
say("注：判定建议（段二胜者 k）——以下为机械汇总，最终以人工复核为准。")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\n[ok] 写出 {OUT.relative_to(ROOT)}")
