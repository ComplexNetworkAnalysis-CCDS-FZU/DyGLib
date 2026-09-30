#!/bin/bash
cd ~/DyGLib
echo "== 任务 712-721 全文 =="
awk 'NR>=712 && NR<=721 {print NR": "length($0)" chars: "substr($0,1,420)}' tools/queue/tasks.txt
echo
echo "== 定位结果目录 =="
ls saved_results/ 2>/dev/null | head
find . -maxdepth 3 -type d -name "*RedditHyperlinkBody*" 2>/dev/null | head -6
echo "== CNS-D RB 训练件是否存在 =="
find . -name "*RedditHyperlinkBody*CNAS-D*.json" -path "*SignLink*" 2>/dev/null | head -8
echo "== CNS-FX 文件全库计数 =="
find . -name "*CNAS-D*G2.EVT.FX.json" 2>/dev/null | sed 's#.*/##' | head -12
find . -name "*CNAS-D*G2.EVT.FX.json" 2>/dev/null | wc -l
echo "== task 719 log 尾 =="
tail -c 800 tools/queue/logs/task_719.log 2>/dev/null
echo
echo "== task 718 log 尾 =="
tail -c 400 tools/queue/logs/task_718.log 2>/dev/null
