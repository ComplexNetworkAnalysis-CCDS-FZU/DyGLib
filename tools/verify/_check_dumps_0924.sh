#!/bin/bash
cd /home/fedsa/DyGLib
echo "== queue.log 尾部 8 行 =="
tail -8 tools/queue/queue.log
echo ""
echo "== dump 输出目录 =="
ls -la results/samples_dump/*/ 2>/dev/null | head -20
echo ""
echo "== 最近 task 日志（含 dump 关键行） =="
for f in $(ls -t tools/queue/logs/task_31*.log tools/queue/logs/task_32*.log 2>/dev/null | head -3); do
  echo "--- $f"
  grep -E "dump_samples|Epoch|final|Error|Traceback" "$f" | tail -5
done
