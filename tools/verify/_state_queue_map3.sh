#!/bin/bash
cd ~/DyGLib/tools/queue
awk 'NR>=390 && NR<=437 {
  ds=""; seed=""; nn=""; lf=""; tag="";
  if (match($0, /--dataset-name [A-Za-z]+/)) { ds=substr($0,RSTART+15,RLENGTH-15) }
  if (match($0, /--seeds [0-9]+/)) { seed=substr($0,RSTART+8,RLENGTH-8) }
  if (match($0, /--num-neighbors [0-9]+/)) { nn=substr($0,RSTART+16,RLENGTH-16) }
  if (match($0, /--common-neighbors-look-forward [0-9]+/)) { lf=substr($0,RSTART+32,RLENGTH-32) }
  if (match($0, /--module-bte-[a-z0-9-]+/)) { tag=substr($0,RSTART,RLENGTH) }
  if (match($0, /--recent-block [0-9]+/)) { tag=tag " RK" substr($0,RSTART+15,RLENGTH-15) }
  if ($0 ~ /train_sign_link_3class/) { task="linksign" } else if ($0 ~ /train_link_sign_prediction/) { task="sign" } else { task="??" }
  printf "%d: %s %s seed%s NN%s LF%s %s\n", NR, task, ds, seed, nn, lf, tag
}' tasks.txt
echo "== 每 tag 总行数与区间 =="
for t in b2-density b4-channel b5-continuous no-module-bte-evidence-gate; do
  echo "$t: $(grep -c -- "--module-$t" tasks.txt) 行; 首行 $(grep -n -- "--module-$t" tasks.txt | head -1 | cut -d: -f1); 末行 $(grep -n -- "--module-$t" tasks.txt | tail -1 | cut -d: -f1)"
done
echo "== 尾部 10 行标记确认 grid =="; tail -n 3 tasks.txt | grep -o "num-neighbors [0-9]* --common-neighbors-look-forward [0-9]*"
