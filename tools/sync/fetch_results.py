"""tools/sync/fetch_results.py — 把服务器上已完成的实验结果 JSON 同步到本地归档。

用法（仓库根目录）：
    python tools/sync/fetch_results.py --set e3      # E-3 修复后 8 个（RB+WV × P{1,3,5,7}）
    python tools/sync/fetch_results.py --set e4      # E-4 修复后 TD ×1（WV NN-15 LF-10）
    python tools/sync/fetch_results.py --set main-a  # 主表 sign RT+RB（10）
    python tools/sync/fetch_results.py --set main-b  # 主表 linksign RT+RB（10，待 task#7 完成）
    python tools/sync/fetch_results.py --set main-c  # 主表 WV sign+linksign（10，待 task#8/#9）
    python tools/sync/fetch_results.py --set main-d  # 主表 sign BA+OTC（10）
    python tools/sync/fetch_results.py --set main-e  # 主表 linksign BA+OTC（10）
    python tools/sync/fetch_results.py --set main-all # 主表全部 50（a+b+c+d+e）
    python tools/sync/fetch_results.py --set e2      # E-2 修复后 20 个（待队列 #14/#15 完成后）
    python tools/sync/fetch_results.py --set e3x     # E-3 扩展：P 补全 BA/OTC/RT（12，R2-11）
    python tools/sync/fetch_results.py --set e2b     # E-2 补：BTE-off 对照 CNAS-only/vanilla ×5（10）
    python tools/sync/fetch_results.py --set e2d     # 补充：sign 邻域网格 RT/RB/BA 各 5 点（15）
    python tools/sync/fetch_results.py --set e5      # E-5 10 个（sign RT 已可；linksign RT 待完整）
    python tools/sync/fetch_results.py --set nh5     # 第 2 批补充：胜者 5 种子 RT/RB + 复核 5 种子 WV20/20·OTC40/15 + OTC/WV 邻域 + RT 探边（32）
    python tools/sync/fetch_results.py --set base-gpu  # Baseline GPU 产物：DySDGNN/DynamiSE 各 15 JSON + 2 summary CSV（32）
    python tools/sync/fetch_results.py --set all     # e2 + e5

纪律（2026-09-11 用户指示）：
- **只读服务器**（ssh 读取；不写/不删/不 scp）；服务器访问须**用户逐次明确许可**。
- 本脚本把「服务器 → 本地 results/**/raw/」的拉取动作固化为可追溯流程：
  单次 ssh 调用（服务器侧 python 打包 base64），本地写入后打印 sha256 清单，
  并向 `results/_sync_raw_log.csv` 追加一行（时间/文件/sha256）。
- 拉取后可用 `python tools/verify/check_archived_results.py` 对照本地归档指纹。
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 仓库根（tools/sync/ 的上两级）
SSH_TARGET = "fedsa@172.17.173.102"
REMOTE_CMD = (
    "cd ~/DyGLib && source /home/fedsa/anaconda3/etc/profile.d/conda.sh "
    "&& conda activate gc && python -"
)

# 每个条目: (remote_glob 模板, local_dir 模板, 数据集列表 or None, 期望文件数)
E2_BASE = "saved_results/SignLinkPrediction/SignDyGFormer/{ds}/*NN-Best.LF-Best.RAS-?.RASE-?.BTE-E.CNAS-E.P1.TE.json"
E5_LINKSIGN = "saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkTitle/*NN-60.LF-1.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
E5_SIGN = "saved_results/LinkSign/SignDyGFormer/RedditHyperlinkTitle/*NN-100.LF-1.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
E3_RB = "saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody/*NN-80.LF-3.RAS-E.RASE-E.BTE-E.CNAS-E.P[1357].TE.json"
E3_WV = "saved_results/SignLinkPrediction/SignDyGFormer/WikiVote/*NN-15.LF-10.RAS-E.RASE-E.BTE-E.CNAS-E.P[1357].TE.json"
E3X_BA = "saved_results/SignLinkPrediction/SignDyGFormer/BitcoinAlpha/SignDyGFormer_seed42.NN-40.LF-15.RAS-E.RASE-E.BTE-E.CNAS-E.P[1357].TE.json"
E3X_OTC = "saved_results/SignLinkPrediction/SignDyGFormer/BitcoinOTC/SignDyGFormer_seed42.NN-80.LF-5.RAS-E.RASE-E.BTE-E.CNAS-E.P[1357].TE.json"
E3X_RT = "saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkTitle/SignDyGFormer_seed42.NN-60.LF-1.RAS-E.RASE-E.BTE-E.CNAS-E.P[1357].TE.json"
E2B = "saved_results/SignLinkPrediction/SignDyGFormer/{ds}/SignDyGFormer_seed42.NN-Best.LF-Best.RAS-D.RASE-D.BTE-D.CNAS-[ED].P1.TE.json"
GRID_RT_A = "saved_results/LinkSign/SignDyGFormer/RedditHyperlinkTitle/SignDyGFormer_seed42.NN-[68]0.LF-[13].RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
GRID_RT_B = "saved_results/LinkSign/SignDyGFormer/RedditHyperlinkTitle/SignDyGFormer_seed42.NN-100.LF-3.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
GRID_RB_A = "saved_results/LinkSign/SignDyGFormer/RedditHyperlinkBody/SignDyGFormer_seed42.NN-[48]0.LF-[13].RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
GRID_RB_B = "saved_results/LinkSign/SignDyGFormer/RedditHyperlinkBody/SignDyGFormer_seed42.NN-60.LF-3.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
GRID_BA_A = "saved_results/LinkSign/SignDyGFormer/BitcoinAlpha/SignDyGFormer_seed42.NN-20.LF-1[05].RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
GRID_BA_B = "saved_results/LinkSign/SignDyGFormer/BitcoinAlpha/SignDyGFormer_seed42.NN-40.LF-10.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
GRID_BA_C = "saved_results/LinkSign/SignDyGFormer/BitcoinAlpha/SignDyGFormer_seed42.NN-60.LF-1[05].RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
E4_TD = "saved_results/SignLinkPrediction/SignDyGFormer/WikiVote/*NN-15.LF-10.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TD.json"
MAIN_SIGN_RT = "saved_results/LinkSign/SignDyGFormer/RedditHyperlinkTitle/*NN-100.LF-1.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
MAIN_SIGN_RB = "saved_results/LinkSign/SignDyGFormer/RedditHyperlinkBody/*NN-60.LF-1.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
MAIN_LINKSIGN_RT = "saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkTitle/*NN-60.LF-1.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
MAIN_LINKSIGN_RB = "saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody/*NN-80.LF-3.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
MAIN_SIGN_WV = "saved_results/LinkSign/SignDyGFormer/WikiVote/*NN-40.LF-15.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
MAIN_LINKSIGN_WV = "saved_results/SignLinkPrediction/SignDyGFormer/WikiVote/*NN-15.LF-10.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
MAIN_SIGN_BA = "saved_results/LinkSign/SignDyGFormer/BitcoinAlpha/*NN-40.LF-15.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
# 2026-09-14 OTC sign 换装（用户拍板）：NN-60/LF-10 → NN-40/LF-15（0.8775±0.0054 vs 旧 0.8696±0.0048，配对 Δ+0.0079/t=3.16）
MAIN_SIGN_OTC = "saved_results/LinkSign/SignDyGFormer/BitcoinOTC/*NN-40.LF-15.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
MAIN_LINKSIGN_BA = "saved_results/SignLinkPrediction/SignDyGFormer/BitcoinAlpha/*NN-40.LF-15.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
MAIN_LINKSIGN_OTC = "saved_results/SignLinkPrediction/SignDyGFormer/BitcoinOTC/*NN-80.LF-5.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"

# ---- 第 2 批补充（2026-09-14）：胜者 5 种子 + OTC/WV 邻域探索 + RT 探边（含补跑 6 点后完整 22）----
WIN_RT_5SEED = "saved_results/LinkSign/SignDyGFormer/RedditHyperlinkTitle/SignDyGFormer_seed*.NN-100.LF-3.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
WIN_RB_5SEED = "saved_results/LinkSign/SignDyGFormer/RedditHyperlinkBody/SignDyGFormer_seed*.NN-40.LF-1.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
# --- #61/#62 复核 5 种子（2026-09-14 晚，用户批准）---
WIN_WV_20_20_5SEED = "saved_results/LinkSign/SignDyGFormer/WikiVote/SignDyGFormer_seed*.NN-20.LF-20.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
WIN_OTC_40_15_5SEED = "saved_results/LinkSign/SignDyGFormer/BitcoinOTC/SignDyGFormer_seed*.NN-40.LF-15.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
NH_OTC_40_5 = "saved_results/LinkSign/SignDyGFormer/BitcoinOTC/SignDyGFormer_seed42.NN-40.LF-5.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
NH_OTC_40_15 = "saved_results/LinkSign/SignDyGFormer/BitcoinOTC/SignDyGFormer_seed42.NN-40.LF-15.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
NH_OTC_80_5 = "saved_results/LinkSign/SignDyGFormer/BitcoinOTC/SignDyGFormer_seed42.NN-80.LF-5.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
NH_OTC_80_15 = "saved_results/LinkSign/SignDyGFormer/BitcoinOTC/SignDyGFormer_seed42.NN-80.LF-15.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
NH_OTC_60_15 = "saved_results/LinkSign/SignDyGFormer/BitcoinOTC/SignDyGFormer_seed42.NN-60.LF-15.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
NH_WV_20_10 = "saved_results/LinkSign/SignDyGFormer/WikiVote/SignDyGFormer_seed42.NN-20.LF-10.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
NH_WV_20_20 = "saved_results/LinkSign/SignDyGFormer/WikiVote/SignDyGFormer_seed42.NN-20.LF-20.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
NH_WV_60_10 = "saved_results/LinkSign/SignDyGFormer/WikiVote/SignDyGFormer_seed42.NN-60.LF-10.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
NH_WV_60_20 = "saved_results/LinkSign/SignDyGFormer/WikiVote/SignDyGFormer_seed42.NN-60.LF-20.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
NH_WV_40_20 = "saved_results/LinkSign/SignDyGFormer/WikiVote/SignDyGFormer_seed42.NN-40.LF-20.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
NH_RT_100_5 = "saved_results/LinkSign/SignDyGFormer/RedditHyperlinkTitle/SignDyGFormer_seed42.NN-100.LF-5.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
NH_RT_120_3 = "saved_results/LinkSign/SignDyGFormer/RedditHyperlinkTitle/SignDyGFormer_seed42.NN-120.LF-3.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"

# ---- ⓑ/E-2c 5 种子加深批（2026-09-14 晚起；命名 = NN-Best.LF-Best + 模块旗标）----
# e2s（ⓑ）= full / vanilla / CNAS-only 各 5 数据集×5 种子；e2c = BTE-only / base 各 25 + RT 的 +RAS/+RAE 各 5
E2S_FULL = "saved_results/SignLinkPrediction/SignDyGFormer/{ds}/SignDyGFormer_seed*.NN-Best.LF-Best.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
E2S_VANILLA = "saved_results/SignLinkPrediction/SignDyGFormer/{ds}/SignDyGFormer_seed*.NN-Best.LF-Best.RAS-D.RASE-D.BTE-D.CNAS-D.P1.TE.json"
E2S_CNASONLY = "saved_results/SignLinkPrediction/SignDyGFormer/{ds}/SignDyGFormer_seed*.NN-Best.LF-Best.RAS-D.RASE-D.BTE-D.CNAS-E.P1.TE.json"
E2C_BTEONLY = "saved_results/SignLinkPrediction/SignDyGFormer/{ds}/SignDyGFormer_seed*.NN-Best.LF-Best.RAS-D.RASE-D.BTE-E.CNAS-D.P1.TE.json"
E2C_BASE = "saved_results/SignLinkPrediction/SignDyGFormer/{ds}/SignDyGFormer_seed*.NN-Best.LF-Best.RAS-D.RASE-D.BTE-E.CNAS-E.P1.TE.json"
E2C_PLUSRAS = "saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkTitle/SignDyGFormer_seed*.NN-Best.LF-Best.RAS-E.RASE-D.BTE-E.CNAS-E.P1.TE.json"
E2C_PLUSRAE = "saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkTitle/SignDyGFormer_seed*.NN-Best.LF-Best.RAS-D.RASE-E.BTE-E.CNAS-E.P1.TE.json"

# ---- Baseline GPU 产物（2026-09-14 二波：#53/#54；路径相对 ~/DyGLib）----
BASE_M5_DYSDGNN = "../DynamiSE_DySDGNN_repro/outputs/DySDGNN/*.json"
BASE_M5_DYNAMISE = "../DynamiSE_DySDGNN_repro/outputs/DynamiSE/*.json"
BASE_M5_SUM_DYSDGNN = "../DynamiSE_DySDGNN_repro/outputs/summary_DySDGNN.csv"
BASE_M5_SUM_DYNAMISE = "../DynamiSE_DySDGNN_repro/outputs/summary_DynamiSE.csv"

SETS = {
    "e2": [
        (E2_BASE, "results/E-2_ablation/raw/{ds}",
         ["BitcoinAlpha", "BitcoinOTC", "WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody"], 20),
    ],
    "e3": [
        (E3_RB, "results/E-3_patch/raw", None, 4),
        (E3_WV, "results/E-3_patch/raw", None, 4),
    ],
    "e3x": [
        (E3X_BA, "results/E-3_patch/raw", None, 4),
        (E3X_OTC, "results/E-3_patch/raw", None, 4),
        (E3X_RT, "results/E-3_patch/raw", None, 4),
    ],
    "e2b": [
        (E2B, "results/E-2_ablation/raw_bte/{ds}",
         ["BitcoinAlpha", "BitcoinOTC", "WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody"], 10),
    ],
    "e2s": [
        (E2S_FULL, "results/E-2_ablation/raw_seeds/{ds}",
         ["BitcoinAlpha", "BitcoinOTC", "WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody"], 25),
        (E2S_VANILLA, "results/E-2_ablation/raw_seeds/{ds}",
         ["BitcoinAlpha", "BitcoinOTC", "WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody"], 25),
        (E2S_CNASONLY, "results/E-2_ablation/raw_seeds/{ds}",
         ["BitcoinAlpha", "BitcoinOTC", "WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody"], 25),
    ],
    "e2c": [
        (E2C_BTEONLY, "results/E-2_ablation/raw_seeds/{ds}",
         ["BitcoinAlpha", "BitcoinOTC", "WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody"], 25),
        (E2C_BASE, "results/E-2_ablation/raw_seeds/{ds}",
         ["BitcoinAlpha", "BitcoinOTC", "WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody"], 25),
        (E2C_PLUSRAS, "results/E-2_ablation/raw_seeds/RedditHyperlinkTitle", None, 5),
        (E2C_PLUSRAE, "results/E-2_ablation/raw_seeds/RedditHyperlinkTitle", None, 5),
    ],
    # 2026-09-15 中期取数（BTE 进度核查）：BA/OTC/RT 已完成的三配置（full/CNAS-only/BTE-only）+ CNAS-only WV
    "e2now": [
        (E2C_BTEONLY, "results/E-2_ablation/raw_seeds/{ds}",
         ["BitcoinAlpha", "BitcoinOTC", "RedditHyperlinkTitle"], 15),
        (E2S_FULL, "results/E-2_ablation/raw_seeds/{ds}",
         ["BitcoinAlpha", "BitcoinOTC", "RedditHyperlinkTitle"], 15),
        (E2S_CNASONLY, "results/E-2_ablation/raw_seeds/{ds}",
         ["BitcoinAlpha", "BitcoinOTC", "WikiVote"], 15),
    ],    # 2026-09-15 进度快照取数（配合 --allow-partial）：full / CNAS-only / BTE-only × 全 5 数据集
    "e2snap": [
        (E2S_FULL, "results/E-2_ablation/raw_seeds/{ds}",
         ["BitcoinAlpha", "BitcoinOTC", "WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody"], 25),
        (E2S_CNASONLY, "results/E-2_ablation/raw_seeds/{ds}",
         ["BitcoinAlpha", "BitcoinOTC", "WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody"], 25),
        (E2C_BTEONLY, "results/E-2_ablation/raw_seeds/{ds}",
         ["BitcoinAlpha", "BitcoinOTC", "WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody"], 25),
    ],    "e2d": [
        (GRID_RT_A, "results/sign_neighborhood/raw/RedditHyperlinkTitle", None, 4),
        (GRID_RT_B, "results/sign_neighborhood/raw/RedditHyperlinkTitle", None, 1),
        (GRID_RB_A, "results/sign_neighborhood/raw/RedditHyperlinkBody", None, 4),
        (GRID_RB_B, "results/sign_neighborhood/raw/RedditHyperlinkBody", None, 1),
        (GRID_BA_A, "results/sign_neighborhood/raw/BitcoinAlpha", None, 2),
        (GRID_BA_B, "results/sign_neighborhood/raw/BitcoinAlpha", None, 1),
        (GRID_BA_C, "results/sign_neighborhood/raw/BitcoinAlpha", None, 2),
    ],
    "e4": [
        (E4_TD, "results/E-4_time_decay/raw", None, 1),
    ],
    "e5": [
        (E5_LINKSIGN, "results/E-5_significance/raw/linksign", None, 5),
        (E5_SIGN, "results/E-5_significance/raw/sign", None, 5),
    ],
    "main-a": [
        (MAIN_SIGN_RT, "results/main_tables/raw/sign/RedditHyperlinkTitle", None, 5),
        (MAIN_SIGN_RB, "results/main_tables/raw/sign/RedditHyperlinkBody", None, 5),
    ],
    "main-b": [
        (MAIN_LINKSIGN_RT, "results/main_tables/raw/linksign/RedditHyperlinkTitle", None, 5),
        (MAIN_LINKSIGN_RB, "results/main_tables/raw/linksign/RedditHyperlinkBody", None, 5),
    ],
    "main-c": [
        (MAIN_SIGN_WV, "results/main_tables/raw/sign/WikiVote", None, 5),
        (MAIN_LINKSIGN_WV, "results/main_tables/raw/linksign/WikiVote", None, 5),
    ],
    "main-d": [
        (MAIN_SIGN_BA, "results/main_tables/raw/sign/BitcoinAlpha", None, 5),
        (MAIN_SIGN_OTC, "results/main_tables/raw/sign/BitcoinOTC", None, 5),
    ],
    "main-e": [
        (MAIN_LINKSIGN_BA, "results/main_tables/raw/linksign/BitcoinAlpha", None, 5),
        (MAIN_LINKSIGN_OTC, "results/main_tables/raw/linksign/BitcoinOTC", None, 5),
    ],
    "nh5": [
        (WIN_RT_5SEED, "results/sign_neighborhood/raw/RedditHyperlinkTitle", None, 5),
        (WIN_RB_5SEED, "results/sign_neighborhood/raw/RedditHyperlinkBody", None, 5),
        (WIN_WV_20_20_5SEED, "results/sign_neighborhood/raw/WikiVote", None, 5),
        (WIN_OTC_40_15_5SEED, "results/sign_neighborhood/raw/BitcoinOTC", None, 5),
        (NH_OTC_40_5, "results/sign_neighborhood/raw/BitcoinOTC", None, 1),
        (NH_OTC_40_15, "results/sign_neighborhood/raw/BitcoinOTC", None, 1),
        (NH_OTC_80_5, "results/sign_neighborhood/raw/BitcoinOTC", None, 1),
        (NH_OTC_80_15, "results/sign_neighborhood/raw/BitcoinOTC", None, 1),
        (NH_OTC_60_15, "results/sign_neighborhood/raw/BitcoinOTC", None, 1),
        (NH_WV_20_10, "results/sign_neighborhood/raw/WikiVote", None, 1),
        (NH_WV_20_20, "results/sign_neighborhood/raw/WikiVote", None, 1),
        (NH_WV_60_10, "results/sign_neighborhood/raw/WikiVote", None, 1),
        (NH_WV_60_20, "results/sign_neighborhood/raw/WikiVote", None, 1),
        (NH_WV_40_20, "results/sign_neighborhood/raw/WikiVote", None, 1),
        (NH_RT_100_5, "results/sign_neighborhood/raw/RedditHyperlinkTitle", None, 1),
        (NH_RT_120_3, "results/sign_neighborhood/raw/RedditHyperlinkTitle", None, 1),
    ],
    "base-gpu": [
        (BASE_M5_DYSDGNN, "results/baseline_m5/raw/DySDGNN", None, 15),
        (BASE_M5_DYNAMISE, "results/baseline_m5/raw/DynamiSE", None, 15),
        (BASE_M5_SUM_DYSDGNN, "results/baseline_m5", None, 1),
        (BASE_M5_SUM_DYNAMISE, "results/baseline_m5", None, 1),
    ],
}
SETS["main-all"] = SETS["main-a"] + SETS["main-b"] + SETS["main-c"] + SETS["main-d"] + SETS["main-e"]
SETS["all"] = SETS["e2"] + SETS["e5"]


def build_globs(specs):
    """展开成 [(glob, local_dir)] 列表。"""
    out = []
    for (gt, lt, datasets, _exp) in specs:
        if datasets:
            for ds in datasets:
                out.append((gt.format(ds=ds), lt.format(ds=ds)))
        else:
            out.append((gt, lt))
    return out


def remote_fetch(globs) -> list:
    """单次 ssh 拉取：返回 [[glob_idx, remote_path, b64], ...]。"""
    code = (
        "import glob, base64, json\n"
        f"globs = {json.dumps([g for g, _ in globs])}\n"
        "out = []\n"
        "for gi, g in enumerate(globs):\n"
        "    for f in sorted(glob.glob(g)):\n"
        "        out.append([gi, f, base64.b64encode(open(f, 'rb').read()).decode()])\n"
        "print('===SYNC_JSON===')\n"
        "print(json.dumps(out))\n"
        "print('===SYNC_END===')\n"
    )
    res = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=15", SSH_TARGET, REMOTE_CMD],
        input=code, capture_output=True, text=True, encoding="utf-8", timeout=300,
    )
    if "===SYNC_JSON===" not in (res.stdout or ""):
        print("[ERR] 服务器无预期输出；stderr 尾部：", file=sys.stderr)
        print((res.stderr or "")[-800:], file=sys.stderr)
        sys.exit(1)
    body = res.stdout.split("===SYNC_JSON===")[1].split("===SYNC_END===")[0].strip()
    return json.loads(body)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SETS), required=True)
    ap.add_argument("--allow-partial", action="store_true",
                    help="允许实际文件数少于期望（进度快照用；逐 glob 打印匹配数）")
    args = ap.parse_args()

    specs = SETS[args.set]
    globs = build_globs(specs)
    expect = sum(e for *_, e in specs)

    print(f"[fetch] set={args.set} 期望 {expect} 个文件；ssh 读取中…")
    items = remote_fetch(globs)
    if len(items) != expect:
        if not args.allow_partial:
            print(f"[ERR] 实际匹配 {len(items)} 个，期望 {expect} —— 模式可能与服务器不符，中止。")
            sys.exit(1)
        print(f"[warn] 实际匹配 {len(items)} 个（期望 {expect}；--allow-partial：按快照收集）")
        from collections import Counter
        cnt = Counter(gi for gi, _, _ in items)
        for gi, (g, ldir) in enumerate(globs):
            print(f"    {ldir:70s} matched={cnt.get(gi, 0)}")

    log_path = ROOT / "results" / "_sync_raw_log.csv"
    new_log = not log_path.exists()
    log_f = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(log_f)
    if new_log:
        writer.writerow(["time", "set", "remote", "local", "sha256", "bytes"])

    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{'数据集/目录':<22s} {'文件':<58s} {'KB':>7s}  sha256[:12]")
    print("-" * 110)
    for gi, rpath, b64 in items:
        _, ldir = globs[gi]
        data = base64.b64decode(b64)
        local = ROOT / ldir / Path(rpath).name
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        dig = sha256_bytes(data)
        writer.writerow([ts, args.set, rpath, str(local.relative_to(ROOT)).replace("\\", "/"), dig, len(data)])
        try:
            auc = json.loads(data.decode("utf-8"))["test metrics"]["auc"]
        except Exception:
            auc = "-"
        print(f"{ldir.split('/')[-1]:<22s} {Path(rpath).name:<58s} {len(data)/1024:7.1f}  {dig[:12]}  auc={auc}")
    log_f.close()
    print(f"[ok] 已写入 {len(items)} 个文件；日志追加于 {log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
