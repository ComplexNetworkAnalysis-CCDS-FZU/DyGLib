#!/usr/bin/env bash
# ============================================================
# SignDyG 修订实验 - 服务器数据就绪检查与补齐脚本
# 用法: bash server_setup.sh [旧副本路径]
#   例: bash server_setup.sh ~/DyGLib_old_backup
# 幂等：可重复执行；缺失的才拷贝/生成。
# ============================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLD_DIR="${1:-}"

# 需要全量文件的签名数据集
FULL_DATASETS=(BitcoinAlpha BitcoinOTC)
# 需要 tail20000 的数据集
TAIL_DATASETS=(WikiVote RedditHyperlinkTitle RedditHyperlinkBody)
# 本轮实验真正用到的数据集
ALL_DATASETS=(BitcoinAlpha BitcoinOTC WikiVote RedditHyperlinkTitle RedditHyperlinkBody)

echo "== [1/5] 代码检查 =="
for f in \
  models/SignDyGFormer.py utils/DataLoader.py utils/load_configs.py utils/noise.py \
  train_link_sign_prediction.py train_sign_link_3class_prediction.py \
  run_experiments.py compute_stats.py preprocess_data/preprocess_data.py; do
  if [ -f "$ROOT/$f" ]; then echo "  [OK] $f"; else echo "  [MISSING] $f"; exit 1; fi
done

echo "== [2/5] DG_data 原始数据检查 =="
for d in "${ALL_DATASETS[@]}"; do
  if [ -f "$ROOT/DG_data/$d/$d.csv" ]; then echo "  [OK] DG_data/$d/$d.csv"; else echo "  [MISSING] DG_data/$d/$d.csv"; fi
done

echo "== [3/5] processed_data 检查 =="
for d in "${FULL_DATASETS[@]}"; do
  for ext in csv npy; do
    if [ -f "$ROOT/processed_data/$d/ml_$d.$ext" ]; then echo "  [OK] ml_$d.$ext"; else echo "  [MISSING] ml_$d.$ext"; fi
  done
done
for d in "${TAIL_DATASETS[@]}"; do
  for ext in csv npy; do
    if [ -f "$ROOT/processed_data/$d/ml_${d}_tail20000.$ext" ]; then
      echo "  [OK] ml_${d}_tail20000.$ext"
    else
      echo "  [MISSING] ml_${d}_tail20000.$ext"
    fi
  done
done

echo "== [4/5] 补齐缺失数据 =="
if [ -n "$OLD_DIR" ] && [ -d "$OLD_DIR" ]; then
  for d in "${ALL_DATASETS[@]}"; do
    if [ ! -d "$ROOT/DG_data/$d" ] && [ -d "$OLD_DIR/DG_data/$d" ]; then
      cp -r "$OLD_DIR/DG_data/$d" "$ROOT/DG_data/" && echo "  从旧副本拷贝 DG_data/$d"
    fi
  done
  if [ ! -d "$ROOT/processed_data" ] && [ -d "$OLD_DIR/processed_data" ]; then
    cp -r "$OLD_DIR/processed_data" "$ROOT/" && echo "  从旧副本拷贝 processed_data"
  fi
else
  echo "  未提供旧副本路径，跳过拷贝（仅检查）"
fi

# 生成缺失的 WikiVote tail20000
if [ ! -f "$ROOT/processed_data/WikiVote/ml_WikiVote_tail20000.csv" ]; then
  if [ -f "$ROOT/DG_data/WikiVote/WikiVote.csv" ]; then
    echo "  生成 WikiVote tail20000 ..."
    ( cd "$ROOT/preprocess_data" && python preprocess_data.py --dataset-name WikiVote --tail-num 20000 )
    echo "  注意: 结尾 check_data 因 DG_data 无 ml_ 文件报错属正常，文件已生成"
  else
    echo "  [WARN] 缺 DG_data/WikiVote/WikiVote.csv，无法生成 tail20000"
  fi
else
  echo "  [OK] WikiVote tail20000 已存在"
fi

echo "== [5/5] 环境 =="
echo "  安装: conda env create -f $ROOT/environment.yaml"
echo "  或按 requirements.txt 安装；torch 按服务器 CUDA 版本单独装，例如:"
echo "    pip install torch --index-url https://download.pytorch.org/whl/cu121"

echo "== 完成 =="
