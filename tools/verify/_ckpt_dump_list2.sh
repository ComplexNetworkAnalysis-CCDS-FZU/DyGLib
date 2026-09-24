#!/bin/bash
# 2026-09-24: D3b 转储行 ckpt 名 + 参数核对（第二版）
cd /home/fedsa/DyGLib
for ds in WikiVote RedditHyperlinkTitle RedditHyperlinkBody BitcoinAlpha BitcoinOTC; do
  d="saved_models/SignLinkPrediction/SignDyGFormer/$ds/SignDyGFormer_seed42"
  echo "===== $ds"
  for tag in "NN-Best.LF-Best.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.pkl" \
             "NN-Best.LF-Best.RAS-E.RASE-E.BTE-D.CNAS-E.P1.TE.pkl"; do
    [ -f "$d/SignDyGFormer_seed42.$tag" ] && echo "  YES  $tag" || echo "  NO   $tag"
  done
done
echo ""
echo "########## param.json 关键字段 ##########"
python3 - <<'PY'
import json, glob
DSS = ["WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody", "BitcoinAlpha", "BitcoinOTC"]
KEYS = ["num_neighbors", "common_neighbors_look_forward", "tail_num", "batch_size",
        "module_balance_theory_encoder", "module_low_ratio_filter",
        "module_repeat_aware_sampler", "module_repeat_aware_sign_encoder",
        "module_common_neighbor_sign_aware_encoder", "module_bte_evidence_gate"]
PATTERNS = [
    ("full ", "NN-Best.LF-Best.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.param.json"),
    ("noBTE", "NN-Best.LF-Best.RAS-E.RASE-E.BTE-D.CNAS-E.P1.TE.param.json"),
    ("G1   ", "NN-*.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.G1.param.json"),
]
for ds in DSS:
    print(f"===== {ds}")
    base = f"saved_models/SignLinkPrediction/SignDyGFormer/{ds}/SignDyGFormer_seed42/SignDyGFormer_seed42."
    for label, pat in PATTERNS:
        files = glob.glob(base + pat)
        if not files:
            print(f"  {label}: (无)")
            continue
        for f in files:
            try:
                j = json.load(open(f))
            except Exception as e:
                print(f"  {label}: 读取失败 {e}")
                continue
            name = f.rsplit("/", 1)[-1].replace("SignDyGFormer_seed42.", "")
            vals = {k: j.get(k) for k in KEYS}
            print(f"  {label} {name}")
            print(f"        {vals}")
PY
