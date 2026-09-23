#!/bin/bash
cd /home/fedsa/DyGLib
source /home/fedsa/anaconda3/etc/profile.d/conda.sh && conda activate gc
echo "=== 队列行数 + 最近调度 ==="
wc -l tools/queue/tasks.txt
tail -6 tools/queue/queue.log
echo "=== 运行中 ==="
ps aux | grep -E "train_(link_sign|sign_link)" | grep -v grep | awk '{print $2, $10}' | head -4
echo ""
echo "=== RLF 对角 vs full（seed42）等价性检查 ==="
python - <<'EOF'
import json, glob
specs = [
    ("WikiVote", 15, 10, 10), ("RedditHyperlinkTitle", 60, 1, 1), ("RedditHyperlinkBody", 80, 3, 3),
]
for ds, nn, lf, kr in specs:
    full = glob.glob(f"saved_results/SignLinkPrediction/SignDyGFormer/{ds}/SignDyGFormer_seed42.NN-{nn}.LF-{lf}.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json")
    diag = glob.glob(f"saved_results/SignLinkPrediction/SignDyGFormer/{ds}/SignDyGFormer_seed42.NN-{nn}.LF-{lf}.RLF-{kr}.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json")
    print(f"--- {ds} (k_r={kr})")
    for tag, files in (("full", full), ("RLF-diag", diag)):
        if not files:
            print(f"  {tag}: 缺 JSON"); continue
        tm = json.load(open(files[0], encoding='utf-8'))['test metrics']
        print(f"  {tag}: auc={tm.get('auc')} f1_wt={tm.get('f1_wt')} f1_mac={tm.get('f1_mac')}")
    # param.json 对照
    fp = f"saved_models/SignLinkPrediction/SignDyGFormer/{ds}/SignDyGFormer_seed42/SignDyGFormer_seed42.NN-{nn}.LF-{lf}.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.param.json"
    dp = f"saved_models/SignLinkPrediction/SignDyGFormer/{ds}/SignDyGFormer_seed42/SignDyGFormer_seed42.NN-{nn}.LF-{lf}.RLF-{kr}.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.param.json"
    import os
    print("  param full:", json.load(open(fp)) if os.path.exists(fp) else "缺失")
    print("  param RLF :", json.load(open(dp)) if os.path.exists(dp) else "缺失")
EOF
