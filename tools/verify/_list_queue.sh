#!/bin/bash
cd /home/fedsa/DyGLib
source /home/fedsa/anaconda3/etc/profile.d/conda.sh && conda activate gc
python - <<'EOF'
import re, pathlib
lines = pathlib.Path('tools/queue/tasks.txt').read_text(encoding='utf-8').splitlines()
log = pathlib.Path('tools/queue/queue.log').read_text(encoding='utf-8', errors='replace')
done = set(int(m) for m in re.findall(r'task#(\d+) running', log))
done |= set(int(m) for m in re.findall(r'task#(\d+) finished', log))
print(f"队列 {len(lines)} 行；已派发 {len(done)} 个任务")
print("=" * 120)
for i, line in enumerate(lines, start=1):
    if i < 190:
        continue
    if line.startswith('-') or line.startswith('#') is False and 'run_experiments' in line:
        label = "GRID(run_experiments 参数网格)"
    else:
        ds = re.search(r'--dataset-name (\S+)', line)
        seeds = re.search(r'--seeds ([\d ]+?)(?: --|$)', line)
        script = "linksign" if 'sign_link_3class' in line else ('sign' if 'link_sign_prediction' in line else '?')
        tags = []
        if '--cnas-tail-fill' in line: tags.append('TF-E')
        m = re.search(r'--recent-block (\d+)', line)
        if m: tags.append(f'RK-{m.group(1)}')
        if '--module-bte-evidence-gate' in line: tags.append('G1')
        if '--eval-only' in line: tags.append('EVT')
        if '--test-thr' in line or '--sign-thr' in line: tags.append('FX')
        label = f"{script:<8} {ds.group(1) if ds else '?':<22} seeds={seeds.group(1) if seeds else '?'} [{' '.join(tags)}]"
    status = "✅已跑" if i in done else ("▶进行" if i in (195, 196) else "待")
    print(f"{i:>4} | {status} | {label}")
EOF
