# -*- coding: utf-8 -*-
"""G2 确认批 val 侧对照表（5 种子）：候选（.G2 新跑）vs 对照（同代际 Full）。

输入：
  results/_val5_20260929.txt        （候选 JSONL 段；含 ssh 工具包头）
  results/_ctrl_val_20260929.jsonl  （对照 JSONL；v4 解析）
输出：results/gconfirm_val_5seed_20260929.txt
"""
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

POINTS = [
    ("linksign", "RedditHyperlinkTitle", 15, 3, 60, 1, "linksign RT 15/3（当前 60/1）", "f1_wt"),
    ("linksign", "RedditHyperlinkBody", 60, 1, 80, 3, "linksign RB 60/1（当前 80/3）", "f1_wt"),
    ("sign", "RedditHyperlinkTitle", 60, 3, 100, 1, "sign RT 60/3（当前 100/1）", "f1_macro"),
    ("sign", "RedditHyperlinkBody", 40, 1, 60, 1, "sign RB 40/1（当前 60/1）", "f1_macro"),
    ("sign", "WikiVote", 15, 10, 40, 15, "sign WV 15/10（当前 40/15）", "f1_macro"),
]


def load_jsonl(path):
    rows = []
    for ln in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if ln.startswith("{") and ln.endswith("}"):
            try:
                d = json.loads(ln)
                if "metrics" in d:
                    rows.append(d)
            except Exception:
                pass
    return rows


cand_rows = [r for r in load_jsonl(ROOT / "results/_val5_20260929.txt") if r.get("note") == "cand"]
ctrl_rows = [r for r in load_jsonl(ROOT / "results/_ctrl_val_20260929.jsonl") if r.get("note") == "ctrl"]


def pick(rows, task, ds, nn, lf, seed):
    for r in rows:
        if r["task"] == task and r["ds"] == ds and r["nn"] == nn and r["lf"] == lf and r["seed"] == seed:
            return r
    return None


lines = []
lines.append("G2 确认批 val 侧对照（5 种子；val 指标 = 训练日志最终 checkpoint epoch 的验证集指标）")
lines.append("主指标：linksign = f1_wt；sign = f1_macro。Δ = 候选 − 对照。")
lines.append("")
for (task, ds, nn, lf, cnn, clf, label, key) in POINTS:
    lines.append(f"===== {label}=====")
    cs, xs = [], []
    for s in (42, 123, 456, 789, 1024):
        rc = pick(cand_rows, task, ds, nn, lf, s)
        rx = pick(ctrl_rows, task, ds, cnn, clf, s)
        if rc is None or rx is None:
            lines.append(f"  seed{s}: 数据缺失（cand={rc is not None}, ctrl={rx is not None}）")
            continue
        c = rc["metrics"].get(key)
        x = rx["metrics"].get(key)
        d = (c - x) * 1000 if (c is not None and x is not None) else None
        thr = rc["metrics"].get("thr_sign", rc["metrics"].get("thr"))
        lines.append(f"  seed{s:<5d} cand={c:.4f}  ctrl={x:.4f}  Δ={d:+6.1f}‰   (cand thr={thr})")
        cs.append(c); xs.append(x)
    if cs and xs:
        dmean = (st.mean(cs) - st.mean(xs)) * 1000
        pos = sum(1 for c, x in zip(cs, xs) if c > x)
        lines.append(f"  → 均值 cand={st.mean(cs):.4f} ctrl={st.mean(xs):.4f} Δ均值={dmean:+.1f}‰ ({pos}/{len(cs)} 正)")
    # 次级指标（auc）
    cs2, xs2 = [], []
    for s in (42, 123, 456, 789, 1024):
        rc = pick(cand_rows, task, ds, nn, lf, s)
        rx = pick(ctrl_rows, task, ds, cnn, clf, s)
        if rc and rx and "auc" in rc["metrics"] and "auc" in rx["metrics"]:
            cs2.append(rc["metrics"]["auc"]); xs2.append(rx["metrics"]["auc"])
    if cs2:
        d2 = (st.mean(cs2) - st.mean(xs2)) * 1000
        lines.append(f"  [auc] 均值 cand={st.mean(cs2):.4f} ctrl={st.mean(xs2):.4f} Δ={d2:+.1f}‰")
    lines.append("")

out = ROOT / "results/gconfirm_val_5seed_20260929.txt"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"[ok] 写出 {out.relative_to(ROOT)}")
