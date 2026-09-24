#!/bin/bash
cd /home/fedsa/DyGLib
echo "== 总行数 =="
wc -l tools/queue/tasks.txt
echo "== 311-316 行开头 130 字符 =="
sed -n '311,316p' tools/queue/tasks.txt | cut -c1-130
echo "== 各行是否含 dump-samples =="
grep -n "dump-samples" tools/queue/tasks.txt | head -3
echo "== 最近完成的行（done 标记） =="
ls -t tools/queue/done/ 2>/dev/null | head -5
echo "== 运行中 python 训练进程 =="
ps aux | grep -E "train_" | grep -v grep | awk '{print $2, $11, $12, $13, $14, $15}' | head -10
