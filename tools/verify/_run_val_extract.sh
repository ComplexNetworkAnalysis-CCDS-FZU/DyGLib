#!/bin/bash
cd ~/DyGLib || exit 1
git pull --ff-only 2>&1 | tail -1
git log -1 --oneline
echo '=== Gate-1 val extraction (grid seed42, 10 runs) ==='
python3 tools/verify/extract_val_from_logs.py
