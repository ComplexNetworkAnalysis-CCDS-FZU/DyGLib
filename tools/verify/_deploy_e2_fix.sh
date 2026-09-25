#!/bin/bash
cd ~/DyGLib
echo "== 备份服务器侧 npz 修改 =="
tar -czf /tmp/bte_sparsity_server_mod_$(date +%s).tgz results/bte_sparsity 2>/dev/null
echo "备份完成: $(ls -t /tmp/bte_sparsity_server_mod_*.tgz | head -1)"
echo "== 让路（丢弃服务器侧修改；本地已入库新版） =="
git checkout -- results/bte_sparsity
echo "== 再次 pull =="
git pull --ff-only 2>&1 | tail -3
echo "== git log -1 =="
git log -1 --oneline
echo "== 关键标记 grep =="
grep -c "e2_self_recent" utils/direct_neighbor_sampler.py utils/load_configs.py train_sign_link_3class_prediction.py train_link_sign_prediction.py
source /home/fedsa/anaconda3/etc/profile.d/conda.sh
conda activate gc
echo "== E2 真实数据冒烟 =="
python tools/verify/_probe_e2_smoke.py 2>&1 | tail -7
echo "== E2 单测 =="
python tools/verify/test_e2_guard.py 2>&1 | tail -5
echo "== CLI 旗标计数 =="
python train_link_sign_prediction.py --help 2>&1 | grep -c "e2-self-recent"
