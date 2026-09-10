#!/bin/bash
# SignDyG 实验队列守护进程（自动把 tasks.txt 中的任务分发到空闲 GPU）
# 停止：pkill -f queue_daemon.sh
# 任务格式：tasks.txt 每行一条 run_experiments.py 的参数（不含 -g，daemon 自动补）

QD=/home/fedsa/DyGLib/tools/queue
TASKS=$QD/tasks.txt
RUNNING=$QD/running.txt
QLOG=$QD/queue.log
LOGD=$QD/logs
mkdir -p "$LOGD"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$QLOG"; }

gpu_mem() { nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null; }

# 判断 GPU 是否忙碌：0=忙 1=空
busy() {
  local lock=/tmp/gpulock.$1
  if [ -f "$lock" ]; then
    local p
    p=$(cat "$lock" 2>/dev/null)
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then return 0; fi
    rm -f "$lock"
  fi
  local used
  used=$(gpu_mem "$1")
  if [ -n "$used" ] && [ "$used" -gt 500 ]; then return 0; fi
  return 1
}

WAIT_EMPTY=0
WAIT_GPU=0
log "daemon start pid=$$"
while true; do
  tot=$(grep -cv '^[[:space:]]*$' "$TASKS" 2>/dev/null)
  [ -z "$tot" ] && tot=0
  done_n=$(wc -l < "$RUNNING" 2>/dev/null)
  [ -z "$done_n" ] && done_n=0
  if [ "$tot" -le "$done_n" ]; then
    if [ "$WAIT_EMPTY" -eq 0 ]; then log "no pending task (done $done_n/$tot) -> idle"; WAIT_EMPTY=1; fi
    sleep 60
    continue
  fi
  WAIT_EMPTY=0

  n=$((done_n + 1))
  cmd=$(grep -v '^[[:space:]]*$' "$TASKS" | sed -n "${n}p")

  gpu=""
  for i in 0 1; do
    if ! busy "$i"; then gpu=$i; break; fi
  done
  if [ -z "$gpu" ]; then
    if [ "$WAIT_GPU" -eq 0 ]; then
      log "task#$n pending, no free GPU (gpu0=$(gpu_mem 0)MiB gpu1=$(gpu_mem 1)MiB) -> waiting"
      WAIT_GPU=1
    fi
    sleep 60
    continue
  fi
  WAIT_GPU=0

  log "dispatch task#$n -> GPU$gpu : $cmd"
  cd /home/fedsa/DyGLib || exit 1
  source /home/fedsa/anaconda3/etc/profile.d/conda.sh
  conda activate gc
  nohup python run_experiments.py $cmd -g "$gpu" > "$LOGD/task_${n}.log" 2>&1 < /dev/null &
  echo $! > "/tmp/gpulock.$gpu"
  sleep 5
  if [ -f "/tmp/gpulock.$gpu" ] && kill -0 "$(cat /tmp/gpulock.$gpu)" 2>/dev/null; then
    echo "$n" >> "$RUNNING"
    log "task#$n running on GPU$gpu pid=$(cat /tmp/gpulock.$gpu)"
  else
    echo "$n" >> "$RUNNING"
    log "WARN task#$n failed to stay alive (see $LOGD/task_${n}.log)"
  fi
  sleep 60
done
