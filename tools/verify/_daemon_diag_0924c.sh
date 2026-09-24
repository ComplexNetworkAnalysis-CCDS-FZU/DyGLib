#!/bin/bash
cd /home/fedsa/DyGLib
echo "== 时间 =="; date '+%m-%d %H:%M:%S'
echo "== daemon 存活? =="
ps -u fedsa -o pid,etime,cmd | grep -E "queue_daemon" | grep -v grep
echo "== queue.log 尾部 10 行（截断） =="
tail -10 tools/queue/queue.log | cut -c1-120
echo "== 每数据集 npz =="
for ds in WikiVote RedditHyperlinkTitle RedditHyperlinkBody BitcoinAlpha BitcoinOTC; do
  echo "$ds: $(ls results/samples_dump/$ds/ 2>/dev/null | wc -l)"
done
echo "== GPU =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "== running.txt =="
wc -l tools/queue/running.txt
echo "== 最新 task 日志文件 =="
ls -t tools/queue/logs/task_3*.log 2>/dev/null | head -3
