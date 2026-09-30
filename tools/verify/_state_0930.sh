#!/bin/bash
cd ~/DyGLib
echo "== 时间 =="; date
echo "== 指针/总行 =="; echo "ptr=$(wc -l < tools/queue/running.txt) total=$(grep -c . tools/queue/tasks.txt)"
echo "== daemon =="; ps aux | grep queue_daemon | grep -v grep | wc -l
echo "== 在跑进程（截断）=="
ps -eo pid,etime,pcpu,args | grep -E "train_(sign_link_3class|link_sign)_prediction|dyg_mamba|triton|runpy" | grep -v grep | cut -c1-200
echo "== GPU =="; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "== 队列 697-716 =="
sed -n '697,716p' tools/queue/tasks.txt | cut -c1-180
echo "== A 件（NN-80.LF-3 BTE-D TF-E.RK-80）=="
D=saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody
ls -l --time-style=+%m-%d_%H:%M $D/SignDyGFormer_seed*.NN-80.LF-3.RAS-E.RASE-E.BTE-D.CNAS-E.P1.TE.TF-E.RK-80.json 2>/dev/null | awk '{print $6, $7}'
echo "A count: $(ls $D/SignDyGFormer_seed*.NN-80.LF-3.RAS-E.RASE-E.BTE-D.CNAS-E.P1.TE.TF-E.RK-80.json 2>/dev/null | wc -l)"
echo "== B 件（NN-60.LF-1 BTE-D G2）=="
ls -l --time-style=+%m-%d_%H:%M $D/SignDyGFormer_seed*.NN-60.LF-1.RAS-E.RASE-E.BTE-D.CNAS-E.P1.TE.G2.json 2>/dev/null | awk '{print $6, $7}'
echo "B count: $(ls $D/SignDyGFormer_seed*.NN-60.LF-1.RAS-E.RASE-E.BTE-D.CNAS-E.P1.TE.G2.json 2>/dev/null | wc -l)"
echo "== CNS 三组 =="
for spec in "SignLinkPrediction RedditHyperlinkTitle 15 3" "SignLinkPrediction RedditHyperlinkBody 60 1" "LinkSign RedditHyperlinkTitle 60 3"; do
  set -- $spec
  n=$(ls saved_results/$1/SignDyGFormer/$2/SignDyGFormer_seed*.NN-$3.LF-$4.RAS-E.RASE-E.BTE-E.CNAS-D.P1.TE.G2.json 2>/dev/null | wc -l)
  echo "$1/$2 NN-$3.LF-$4 CNS-D: $n/5"
done
echo "== mamba 产物 =="
echo "mamba json: $(ls /home/fedsa/DynamiSE_DySDGNN_repro/outputs/DyG-Mamba/*.json 2>/dev/null | wc -l)"
ls -l --time-style=+%m-%d_%H:%M /home/fedsa/DynamiSE_DySDGNN_repro/outputs/DyG-Mamba/*.json 2>/dev/null | tail -6 | awk '{print $6, $7}'
echo "== 近 6 小时新 json（tail 12）=="
find saved_results -name "*.json" ! -name "*profiler*" -mmin -360 | sort | tail -12
echo "== queue.log 尾 6 =="
tail -6 tools/queue/queue.log
