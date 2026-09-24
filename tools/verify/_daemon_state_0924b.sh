#!/bin/bash
cd /home/fedsa/DyGLib
echo "== queue.log 尾部 20 行 =="
tail -20 tools/queue/queue.log 2>/dev/null
echo ""
echo "== 全部 fedsa python 进程 =="
ps -u fedsa -o pid,etime,cmd | grep -E "python|queue" | grep -v grep | head -12
echo ""
echo "== daemon 脚本? =="
ls -la tools/queue/ | head -20
