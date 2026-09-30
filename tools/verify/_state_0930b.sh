#!/bin/bash
cd ~/DyGLib
echo "== 派发历史（696-706）=="
grep -nE "task#(69[6-9]|70[0-6])" tools/queue/queue.log | tail -40
echo "== task logs 列表 =="
ls -l --time-style=+%m-%d_%H:%M tools/queue/logs/ | tail -18
echo "== task logs 内容 699-704 =="
for i in 699 700 701 702 703 704; do
  f=tools/queue/logs/task_$i.log
  echo "--- task_$i: $(wc -c <"$f" 2>/dev/null) bytes"
  head -c 1200 "$f" 2>/dev/null
  echo
done
echo "== 完整队列行 697-711 =="
sed -n '697,711p' tools/queue/tasks.txt
echo "== running.txt 尾 6 =="
tail -6 tools/queue/running.txt | cut -c1-140
