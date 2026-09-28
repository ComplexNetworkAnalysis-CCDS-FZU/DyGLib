# -*- coding: utf-8 -*-
"""把 extract_val_from_logs 的 JSONL 输出整理成表格（本地辅助）。"""
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
rows = []
for ln in src.read_text(encoding="utf-8", errors="replace").splitlines():
    ln = ln.strip()
    if ln.startswith("{") and ln.endswith("}"):
        try:
            rows.append(json.loads(ln))
        except Exception:
            pass
print("parsed rows:", len(rows))
for r in rows:
    m = r.get("metrics", {})
    st = (r.get("save_time") or "")[:16]
    log = r["log"].split("/")[-1][:18]
    print(f"{r['task']:8s} {r['ds'][:14]:14s} NN-{r['nn']:<4}LF-{r['lf']:<3}[{r['note']}] "
          f"saves={r['n_saves']} {st:16s} "
              f"f1_wt={m.get('f1_wt',''):<7} f1_mac={m.get('f1_mac',m.get('f1_macro',''))!s:<7} "
              f"f1_bin={m.get('f1_binary',''):<7} auc={m.get('auc',''):<7} ap={m.get('ap',''):<7} "
              f"thr={m.get('thr',m.get('thr_sign',''))} log={log}")
