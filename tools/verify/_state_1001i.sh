#!/bin/bash
cd ~/DyGLib
date "+%m-%d %H:%M"
echo "== 队列 =="
wc -l < tools/queue/tasks.txt; wc -l < tools/queue/running.txt
tail -5 tools/queue/queue.log | cut -c1-100
echo "== task 733-735 尾 =="
for i in 733 734 735; do echo "--- $i: $(wc -c < tools/queue/logs/task_$i.log 2>/dev/null) bytes"; tail -4 tools/queue/logs/task_$i.log 2>/dev/null | cut -c1-150; done
echo "== ScaDyG 结果文件 =="
ls -l --time-style=+%m-%d_%H:%M ~/DynamiSE_DySDGNN_repro/outputs/ScaDyG/ 2>/dev/null | tail -18
echo "== 网格图 =="
ls -l --time-style=+%m-%d_%H:%M figures/fig_grid_* 2>/dev/null | cut -c1-110
sha256sum figures/fig_grid_heatmap.csv 2>/dev/null
echo "== 网格图生成脚本/来源线索 =="
grep -l "fig_grid" tools/*.py tools/**/*.py 2>/dev/null | head -5
grep -rn "fig_grid_heatmap" --include=*.py . 2>/dev/null | head -5
