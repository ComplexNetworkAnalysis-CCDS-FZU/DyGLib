#!/bin/bash
# 2026-09-24: 服务器侧部署与校验（D3b）
cd /home/fedsa/DyGLib
echo "== 远端与分支 =="
git remote -v
git branch --show-current
echo "== 当前 HEAD =="
git log --oneline -1
echo "== 拉取 =="
git fetch origin
git pull --ff-only
echo "== 拉取后 HEAD =="
git log --oneline -1
echo "== 校验新参数 =="
grep -n "dump-samples\|dump_samples" utils/load_configs.py | head -5
grep -n "dump_samples" evaluate_models_utils.py | head -8
echo "== 工作区干净度 =="
git status --short | head -5
