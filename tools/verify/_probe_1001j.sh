#!/bin/bash
cd ~/DyGLib
echo "== 历史 scadyg 队列行（env 线索）=="
grep -n "train_sign_scadyg\|scadyg" tools/queue/tasks.txt | head -6 | cut -c1-230
echo "== 历史队列备份里的 scadyg 行 =="
ls tools/queue/tasks.txt.bak-* 2>/dev/null | tail -3
for b in $(ls -t tools/queue/tasks.txt.bak-* 2>/dev/null | head -6); do
  grep -l "scadyg" "$b" 2>/dev/null
done | head -3
echo "== conda envs =="
source /home/fedsa/anaconda3/etc/profile.d/conda.sh
conda env list | sed 's/^#.*//' | grep -v '^$'
echo "== 各 env 是否有 dgl =="
for e in gc dygmamba scadyg base torch_venv; do
  printf "  %-12s " "$e"
  conda run -n "$e" python -c "import dgl,torch; print('dgl',dgl.__version__,'torch',torch.__version__)" 2>&1 | tail -1 | cut -c1-80
done
