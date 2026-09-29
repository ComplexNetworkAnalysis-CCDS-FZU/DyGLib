#!/bin/bash
cd ~/DyGLib || exit 1
git pull --ff-only 2>&1 | tail -1
git log -1 --oneline
echo '=== control val (5 seeds, newest-only; v4 parser) ==='
python3 tools/verify/extract_val_from_logs.py --spec-file /tmp/ctrl_specs.txt --newest-only | tee /tmp/ctrl_val.jsonl
