#!/bin/bash
cd ~/DyGLib || exit 1
echo '== disk =='
df -h ~ | tail -2
echo '== log sizes 699-705 =='
ls -l --time-style=+%m-%d_%H:%M tools/queue/logs/ | grep -E "task_(699|70[0-5])\.log"
echo
for i in 700 701 702 703; do
  echo "===== task_$i.log tail 40 ====="
  tail -40 "tools/queue/logs/task_$i.log" 2>/dev/null | cut -c1-200
  echo
done
