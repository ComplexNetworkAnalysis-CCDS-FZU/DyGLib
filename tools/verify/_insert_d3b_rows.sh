#!/bin/bash
cd /home/fedsa/DyGLib
python3 tools/queue/edit_remote_tasks.py --insert-after 313 --lines-file tools/queue/insert_d3b_20260924.txt
echo "== 插入后总行数 =="
wc -l tools/queue/tasks.txt
echo "== 314 / 315 / 328 / 329 / 330 行（前缀120） =="
sed -n '314p;315p;328p;329p;330p' tools/queue/tasks.txt | cut -c1-120
echo "== dump-samples 行号 =="
grep -n "dump-samples" tools/queue/tasks.txt | cut -d: -f1 | tr '\n' ' '
echo ""
