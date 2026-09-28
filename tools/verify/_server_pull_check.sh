#!/bin/bash
cd ~/DyGLib || exit 1
git pull --ff-only 2>&1 | tail -2
git log -1 --oneline
source /home/fedsa/anaconda3/etc/profile.d/conda.sh
conda activate gc
python - <<'EOF'
from utils.load_configs import SignPredictArgs
n = SignPredictArgs(grid_confirm_g2=True, num_neighbors=15, common_neighbors_look_forward=3,
                    module_repeat_aware_sampler=True, module_repeat_aware_sign_encoder=True,
                    save_model_name='SignDyGFormer_seed42').result_save_name
print('SERVER G2 NAME:', n)
assert n == 'SignDyGFormer_seed42.NN-15.LF-3.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.G2'
print('SERVER G2 OK')
EOF
