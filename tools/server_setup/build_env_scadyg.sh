#!/bin/bash
# 服务器端 env 构建：scadyg（R1-4 波二 · ScaDyG 旧栈；2026-09-14 用户批准，Code 执行）
# 运行（服务器）：cd ~/DyGLib && nohup bash tools/server_setup/build_env_scadyg.sh > ~/envbuild_scadyg.log 2>&1 &
# 说明：按 Baseline 验证过的组合装（torch1.12.1+cu116 / torch-geometric 2.5.3 / dgl 1.0.0 / deepsnap / py-tgb）。
set -x
source /home/fedsa/anaconda3/etc/profile.d/conda.sh
CONDA=/home/fedsa/anaconda3/bin/conda

echo "[1] 创建 env scadyg (python 3.9)"
$CONDA env list | grep -qw scadyg || $CONDA create -y -n scadyg python=3.9
conda activate scadyg
python -V

echo "[2] torch 1.12.1+cu116"
pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 || pip install torch==1.12.1 || exit 21

echo "[3] PyG 全家桶（匹配轮子）"
pip install torch-geometric==2.5.3 || exit 22
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.1+cu116.html || exit 23

echo "[4] 图/数据栈"
pip install dgl==1.0.0 || pip install dgl==1.0.0 -f https://data.dgl.ai/wheels/repo.html || pip install dgl==1.0.0 -f https://data.dgl.ai/wheels/cu116/repo.html || exit 24
pip install deepsnap==0.2.1 py-tgb==0.9.2 || exit 25

echo "[5] 数值/科学计算"
pip install numpy==1.23.4 pandas==1.5.3 scipy==1.9.3 scikit-learn==1.5.0 einops reformer-pytorch==1.4.4 tqdm || exit 26

echo "[6] 自检"
python - <<'PY'
import torch
print('torch', torch.__version__, 'avail', torch.cuda.is_available())
import dgl, deepsnap, tgb, torch_geometric
print('dgl', dgl.__version__, 'pyg', torch_geometric.__version__)
import torch_scatter, torch_sparse
print('torch_scatter/torch_sparse ok')
PY
rc=$?
echo "=== BUILD_SCADYG_DONE rc=$rc ==="
