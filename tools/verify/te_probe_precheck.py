"""te_probe_precheck.py — §4.5 探针前置留档（Code 2026-10-02）。

在启动 3 策略对照前，先把**主口径 sign 归档**（TE/TD 臂会写同名文件）逐件 sha256 + 逐种子值留档，
以便：① 若 TE 臂重跑覆盖同名文件，仍可核对"是否逐位一致"；② 交付时能证明 TE 臂 = 主口径。
输出：results/te_probe_precheck_20261002.txt
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "te_probe_precheck_20261002.txt"
DS = ["BitcoinAlpha", "BitcoinOTC", "WikiVote"]
SEEDS = [42, 123, 456, 789, 1024]

L: list[str] = []
w = L.append
w("§4.5 三策略探针 · 前置留档（启动前快照，Code 2026-10-02）")
w("说明：主口径 sign 归档 = `results/sign_valthr/raw/{ds}/SignDyGFormer_seed*.NN-40.LF-15...TE.json`")
w("     TE 臂重跑会用**同名文件**（同 seed/同配置）；TD/TD-LIN 臂加标签（`.TD` / `.TD-LIN`）⇒ 不冲突。")
w("")
for ds in DS:
    w(f"-- {ds}")
    for s in SEEDS:
        p = ROOT / "results/sign_valthr/raw" / ds / f"SignDyGFormer_seed{s}.NN-40.LF-15.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
        if not p.exists():
            w(f"   seed{s:<5} [missing]")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        m = d.get("test metrics", {})
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        w(f"   seed{s:<5} sha256={h[:16]}  auc={m.get('auc')} f1_macro={m.get('f1_macro')} "
          f"f1_binary={m.get('f1_binary')} acc={m.get('acc')}")
OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
print(f"[ok] {OUT.relative_to(ROOT)}")
