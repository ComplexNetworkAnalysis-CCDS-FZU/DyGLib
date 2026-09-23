#!/bin/bash
# 从 checkpoint param.json 收集：linksign(full/TF-E) 与 sign(full/TF-E) 的阈值
cd /home/fedsa/DyGLib
source /home/fedsa/anaconda3/etc/profile.d/conda.sh
conda activate gc
PY='import json,os,glob
linksign={"WikiVote":(15,10),"RedditHyperlinkTitle":(60,1),"RedditHyperlinkBody":(80,3),"BitcoinAlpha":(40,15),"BitcoinOTC":(80,5)}
sign={"WikiVote":(40,15),"RedditHyperlinkTitle":(100,1),"RedditHyperlinkBody":(60,1),"BitcoinAlpha":(40,15),"BitcoinOTC":(40,15)}
seeds=[42,123,456,789,1024]
out={}
for task,tab,root in (("linksign",linksign,"SignLinkPrediction"),("sign",sign,"LinkSign")):
    for ds,(nn,lf) in tab.items():
        for s in seeds:
            base=f"SignDyGFormer_seed{s}.NN-{nn}.LF-{lf}.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE"
            for tag in ("","TF-E"):
                name=base+(".TF-E" if tag else "")
                p=f"saved_models/{root}/SignDyGFormer/{ds}/SignDyGFormer_seed{s}/{name}.param.json"
                if os.path.exists(p):
                    d=json.load(open(p))
                    key=f"{task}|{ds}|{s}|" + (tag if tag else "full")
                    out[key]=d
print(json.dumps(out,ensure_ascii=False))'
python -c "$PY"
