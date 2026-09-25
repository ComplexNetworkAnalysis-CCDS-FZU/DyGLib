#!/bin/bash
cd ~/DyGLib
echo "== 日志目录结构 =="; ls expm-2026-09-25-logs/ 2>/dev/null | head
echo "== 找 B2/B4/B5/tag 结果文件（近 10 小时）=="
find . -path ./node_modules -prune -o -name "*.json" -mmin -600 -print 2>/dev/null | grep -E "B2|B4|B5|NCN|noBTE|no-bte" | sort | tail -20
echo "== 近 10 小时所有新 json（前 30）=="
find . -name "*.json" -mmin -600 2>/dev/null | grep -v "\.git" | sort | tail -30
echo "== tasks.txt 395-437 明细 =="
awk 'NR>=395 && NR<=437 {
  ds=""; seed=""; nn=""; lf=""; tag="";
  if (match($0, /--dataset-name [A-Za-z]+/)) { ds=substr($0,RSTART+15,RLENGTH-15) }
  if (match($0, /--seeds [0-9]+/)) { seed=substr($0,RSTART+8,RLENGTH-8) }
  if (match($0, /--num-neighbors [0-9]+/)) { nn=substr($0,RSTART+16,RLENGTH-16) }
  if (match($0, /--common-neighbors-look-forward [0-9]+/)) { lf=substr($0,RSTART+32,RLENGTH-32) }
  if (match($0, /--module-bte-[a-z0-9-]+/)) { tag=substr($0,RSTART,RLENGTH) }
  if (match($0, /train_sign_link_3class/)) { task="linksign" } else { task="sign" }
  printf "%d: %s %s seed%s NN%s LF%s %s\n", NR, task, ds, seed, nn, lf, tag
}' tasks.txt
