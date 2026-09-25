#!/bin/bash
cd ~/DyGLib/tools/queue
echo "== 时间 =="; date
echo "== 指针 =="; wc -l < running.txt
echo "== 428-442 行标记 =="
awk 'NR>=428 && NR<=442 {
  ds=""; seed=""; nn=""; lf=""; tag="";
  if (match($0, /--dataset-name [A-Za-z]+/)) { ds=substr($0,RSTART+15,RLENGTH-15) }
  if (match($0, /--seeds [0-9 ]+/)) { seed=substr($0,RSTART+8,RLENGTH-8) }
  if (match($0, /--num-neighbors [0-9]+/)) { nn=substr($0,RSTART+16,RLENGTH-16) }
  if (match($0, /--common-neighbors-look-forward [0-9]+/)) { lf=substr($0,RSTART+32,RLENGTH-32) }
  if (match($0, /--module-bte-[a-z0-9-]+/)) { tag=substr($0,RSTART,RLENGTH) }
  if (match($0, /--e2-self-recent [0-9]+/)) { tag=tag " " substr($0,RSTART,RLENGTH) }
  if ($0 ~ /train_sign_link_3class/) { task="linksign" } else if ($0 ~ /train_link_sign_prediction/) { task="sign" } else { task="??" }
  printf "%d: %s %s seed[%s] NN%s LF%s %s\n", NR, task, ds, seed, nn, lf, tag
}' tasks.txt
echo "== 网格 LF 档确认（468 起）=="
sed -n '468,492p' tasks.txt | grep -o -- "--num-neighbors [0-9]* --common-neighbors-look-forward [0-9]*" | head -30
cd ~/DyGLib
echo "== B2/B4/B5 文件计数 =="
for tag in B2 B4 B5; do
  n=$(find saved_results -name "*.$tag.json" | wc -l)
  echo "$tag: $n"
done
echo "== 近 30 分钟新文件 =="
find saved_results -name "*.json" -mmin -30 | head -10
