#!/bin/bash
cd /home/fedsa/DyGLib
source /home/fedsa/anaconda3/etc/profile.d/conda.sh && conda activate gc
python - <<'EOF'
import json, glob
evt = glob.glob('saved_results/SignLinkPrediction/SignDyGFormer/WikiVote/*EVT*.json')[0]
e1a = glob.glob('saved_results/SignLinkPrediction/SignDyGFormer/WikiVote/*seed123*TF-E.json')
e1a = [p for p in e1a if 'EVT' not in p and 'profiler' not in p][0]
for tag, p in (('E1a auto-thr', e1a), ('E1a @full-thr(0.65/0.01)', evt)):
    d = json.load(open(p, encoding='utf-8'))
    tm = d['test metrics']
    keys = ['auc', 'f1_wt', 'f1_mac', 'ap', 'thr_sign', 'thr_exist']
    print(tag, {k: tm.get(k) for k in keys}, '| eval_only:', d.get('eval_only'), '| fixed_thr:', d.get('fixed_thr'))
EOF
