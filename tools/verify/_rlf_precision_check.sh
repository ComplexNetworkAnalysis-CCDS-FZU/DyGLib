#!/bin/bash
cd /home/fedsa/DyGLib
source /home/fedsa/anaconda3/etc/profile.d/conda.sh && conda activate gc
python - <<'EOF'
import json, glob
specs = [("WikiVote", 15, 10, 10), ("RedditHyperlinkTitle", 60, 1, 1), ("RedditHyperlinkBody", 80, 3, 3)]
keys = ["auc", "f1_wt", "f1_mac", "ap", "f1_mic", "sign_f1", "acc"]
allok = True
for ds, nn, lf, kr in specs:
    f = glob.glob(f"saved_results/SignLinkPrediction/SignDyGFormer/{ds}/SignDyGFormer_seed42.NN-{nn}.LF-{lf}.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json")[0]
    d = glob.glob(f"saved_results/SignLinkPrediction/SignDyGFormer/{ds}/SignDyGFormer_seed42.NN-{nn}.LF-{lf}.RLF-{kr}.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json")[0]
    tf = json.load(open(f, encoding='utf-8'))['test metrics']
    td = json.load(open(d, encoding='utf-8'))['test metrics']
    diff = {k: (tf.get(k), td.get(k)) for k in keys if tf.get(k) != td.get(k)}
    print(f"{ds}: 全键逐位一致={not diff}" + ("" if not diff else f" 差异={diff}"))
    allok &= not diff
print("结论:", "RLF 对角（k_r=k_c）与 full 逐位一致（全部键）" if allok else "存在差异，需重训")
EOF
