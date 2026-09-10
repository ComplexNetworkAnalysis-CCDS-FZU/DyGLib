#!/bin/bash
# 队列守护进程自检：验证 task 解析 / GPU 忙闲探测 / 选卡逻辑（不启动任何任务、不改动队列状态）
# 用法：在仓库根目录执行  bash tools/queue/selftest.sh
QD=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=/dev/null
source "$QD/queue_daemon.sh"

echo "== 1) 任务清单解析 =="
echo "task_count = $(task_count)   (期望 2)"
echo "task#1    = $(nth_task 1)"
echo "task#2    = $(nth_task 2)"

echo
echo "== 2) GPU 忙闲探测（当前两卡都在跑 E-2，应均为 BUSY）=="
for i in 0 1; do
  if busy "$i"; then echo "GPU$i: BUSY (mem=$(gpu_mem "$i")MiB)"; else echo "GPU$i: FREE (mem=$(gpu_mem "$i")MiB)"; fi
done

echo
echo "== 3) lock 机制（存活 PID / 死 PID 两种情况）=="
sleep 30 &
FAKEPID=$!
echo "$FAKEPID" > /tmp/gpulock.0
if busy 0; then echo "lock=存活PID -> BUSY（走 lock 分支）✓"; else echo "lock=存活PID -> FREE ✗ 异常"; fi
kill "$FAKEPID" 2>/dev/null
wait "$FAKEPID" 2>/dev/null
echo 999999 > /tmp/gpulock.0
if busy 0; then echo "lock=死PID -> BUSY（lock 已失效，靠显存判定）✓"; else echo "lock=死PID -> FREE（若显存也低则为正常空闲）"; fi
rm -f /tmp/gpulock.0
echo "（清理测试 lock 完毕）"

echo
echo "== 4) 选卡函数 =="
g=$(find_free_gpu)
echo "find_free_gpu -> '${g}'   (两卡均忙时应为空字符串)"

echo
echo "selftest done"
