#!/bin/bash
cd ~/DyGLib
echo "== 时间 =="; date
echo "== 指针 =="; wc -l < tools/queue/running.txt
echo "== 在跑进程 =="
ps aux | grep -E "train_(sign_link_3class|link_sign)_prediction" | grep -v grep | awk '{for(i=11;i<=NF;i++){if($i ~ /--dataset-name/){ds=$(i+1)}; if($i ~ /--seeds/){sd=$(i+1)}; if($i ~ /--seeds$/){}}; print $2, ds}' | sort | uniq -c
echo "== GPU =="; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "== 各批文件计数与最新时间 =="
for tag in B2 B4 B5; do
  n=$(find saved_results -name "*.$tag.json" | wc -l)
  latest=$(find saved_results -name "*.$tag.json" -printf "%T+ %p\n" | sort | tail -1 | cut -c1-19,21-)
  echo "$tag: $n 个；最新: $latest"
done
echo "== 近 20 分钟新文件 =="
find saved_results -name "*.json" -mmin -20 | grep -v profiler | head -8
echo "== 438-467 行（E2）与 468 起（网格）抽查 =="
sed -n '438p;467p;468p;469p' tools/queue/tasks.txt | cut -c1-200
