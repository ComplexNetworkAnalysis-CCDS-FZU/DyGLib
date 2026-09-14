#!/bin/bash
# 服务器端 env 构建：dygmamba（R1-4 波二 · DyG-Mamba；2026-09-14 用户批准，Code 执行）
# 运行（服务器）：cd ~/DyGLib && nohup bash tools/server_setup/build_env_dygmamba.sh > ~/envbuild_dygmamba.log 2>&1 &
# 说明：服务器无系统 nvcc；先尝试预编译轮子，失败则用 conda 版 cuda-nvcc=11.8 源码构建
#       （CUDA_HOME=$CONDA_PREFIX + --no-build-isolation）。
set -x
source /home/fedsa/anaconda3/etc/profile.d/conda.sh
CONDA=/home/fedsa/anaconda3/bin/conda

echo "[0] conda/pip 配置"
$CONDA config --show channels || true
pip config list || true

echo "[1] 创建 env dygmamba (python 3.10)"
$CONDA env list | grep -qw dygmamba || $CONDA create -y -n dygmamba python=3.10
conda activate dygmamba
python -V

echo "[2] torch 2.1.0（官方 cu118 源 → 失败回退默认源）"
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118 || pip install torch==2.1.0 || exit 11

echo "[3] 基础依赖（v2：pin numpy<2 + 老 setuptools，修复 torch2.1 的 numpy2/pkg_resources 问题）"
pip install "numpy==1.26.4" "setuptools==69.5.1" wheel ninja packaging || exit 12
pip install einops pandas scipy scikit-learn tqdm tabulate || exit 12
python -c "import numpy, torch; print('sanity', numpy.__version__, torch.__version__, float(torch.rand(2).sum()))" || exit 12

echo "[4] mamba 内核（v3：服务器 GitHub 不可达 → 禁用官方 release 轮子探测，强制源码编译）"
[ -x "$CONDA_PREFIX/bin/nvcc" ] || $CONDA install -y -n dygmamba -c nvidia cuda-nvcc=11.8 cuda-cudart-dev=11.8 || exit 13
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="7.5"
export MAX_JOBS=6
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAMBA_FORCE_BUILD=TRUE
pip install --no-build-isolation causal-conv1d==1.4.0 mamba-ssm==2.2.2 || pip install --no-build-isolation causal-conv1d mamba-ssm || exit 14

echo "[5] 自检（import + CUDA 算子冒烟）"
python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())
import causal_conv1d, mamba_ssm, einops
print('causal_conv1d ok; mamba_ssm', getattr(mamba_ssm, '__version__', '?'))
from causal_conv1d import causal_conv1d_fn
x = torch.randn(1, 64, 16, device='cuda'); w = torch.randn(64, 1, 4, device='cuda')
print('conv1d op ok', tuple(causal_conv1d_fn(x, w).shape))
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
u = torch.randn(1, 64, 16, device='cuda'); dt = torch.rand(1, 64, 16, device='cuda') + 1
A = -torch.rand(64, 2, device='cuda'); B = torch.randn(1, 2, 16, device='cuda'); C = torch.randn(1, 2, 16, device='cuda')
D = torch.randn(64, device='cuda')
print('selective_scan ok', tuple(selective_scan_fn(u, dt, A, B, C, D).shape))
PY
rc=$?
echo "=== BUILD_DYGMAMBA_DONE rc=$rc ==="
