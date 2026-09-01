# SignDyG 修订实验运行计划（9 月执行）

> 目标：完成 E-1~E-5（P0-P4），E-6/E-7 已决定跳过。
> 服务器预计 9 月初恢复。**所有命令在 `d:\codes\DyGLib` 根目录执行**（conda 环境激活后）。
> 时间估算是粗估，**以 E-1 实测为准**（RedditTitle@20000 单次 run 预估约 12 小时）。

## 0. 环境准备（服务器开机后一次性）

```bash
conda env create -f environment.yaml          # 或按 requirements.txt 手动装
conda activate signdyg
# torch 按服务器 CUDA 版本单独安装，例如：
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 1. 执行顺序（固定，不要跳步）

### E-1 效率数据（P0）—— 并入 E-5（sign 任务 seed42），无需单独 run
RedditTitle@20000 完整模型（**修复后代码**），记录 4 项：训练总时间 / 推理时间 / 参数量 / 峰值显存
（E-5 中 sign 任务的 seed42 结果 JSON 已含全部 4 项，直接提取即可）

### E-2 增量式消融（P1，5 组 × 2 数据集 = 10 次 run）
任务：**link&sign**（3 分类）；数据集：RedditTitle@20000 + WikiVote@20000；模型：仅 SignDyGFormer

```bash
python run_experiments.py -s linksign -t ablation -m SignDyGFormer \
    -r BitcoinAlpha BitcoinOTC RedditHyperlinkBody -g 0
```
5 组配置（RAS,RAE,BTE,CNAS）：基线 → +CNAS → +RAS → +RAE → +BTE(全开)

### E-3 Patch size 消融（P2，2 数据集 × 4 P = 8 次 run）
数据集：WikiVote@20000 + RedditHyperlinkBody@20000；P ∈ {1, 3, 5, 7}；其余超参用最佳配置

```bash
python run_experiments.py -s linksign -t patch -m SignDyGFormer \
    -r BitcoinAlpha BitcoinOTC RedditHyperlinkTitle -g 0
```

### E-4 时序对比（P3，2 次 run）
对比：时间编码（当前，TE） vs 时间衰减（TD，指数衰减 e^{-λΔt}）；数据集 WikiVote@20000；link&sign 任务

```bash
# A) 时间编码（基线，不加 time-decay 参数即可）
python train_sign_link_3class_prediction.py --dataset-name WikiVote --model SignDyGFormer \
    --seeds 42 --batch-size 200 --num-neighbors 15 --common-neighbors-look-forward 10 \
    --tail-num 20000 --early-stop-notice f1_wt f1_mic ap f1_mac auc -g 0

# B) 时间衰减（λ=1.0，gap_mode 默认 staleness=A；可选 gap=B）
python train_sign_link_3class_prediction.py --dataset-name WikiVote --model SignDyGFormer \
    --seeds 42 --batch-size 200 --num-neighbors 15 --common-neighbors-look-forward 10 \
    --tail-num 20000 --early-stop-notice f1_wt f1_mic ap f1_mac auc \
    --time-decay-lambda 1.0 --time-decay-gap-mode staleness -g 0
```
说明：`--time-decay-lambda` 不提供 = 时间编码（结果文件带 `.TE` 标记）；提供 = 时间衰减（`.TD` 标记），两者不会互相覆盖。

### E-5 统计显著性（P4，双任务 × 5 种子 = 10 次 run）
RedditTitle@20000 完整模型（**修复后代码**），sign 与 link&sign **各 5 个种子**（42,123,456,789,1024）。
既满足审稿人 R2 #8"Reddit Sign Prediction 和 all Link&Sign metrics"都要 p 值的要求，也产出 RedditTitle 修复后主表数据（E-1 效率取自 sign 的 seed42）。

```bash
# sign 任务（5 种子）
python run_experiments.py -s sign -t main -m SignDyGFormer \
    -r BitcoinAlpha BitcoinOTC RedditHyperlinkBody WikiVote -g 0 -e

# link&sign 任务（5 种子）
python run_experiments.py -s linksign -t main -m SignDyGFormer \
    -r BitcoinAlpha BitcoinOTC RedditHyperlinkBody WikiVote -g 0 -e
```
结果汇总 + 配对 t 检验 p 值（对比基线模型需其 5 种子结果文件已存在）：
```bash
python compute_stats.py --task sign --dataset RedditHyperlinkTitle --model SignDyGFormer \
    --pattern "RAS-E.RASE-E.BTE-E.CNAS-E.TE"
python compute_stats.py --task linksign --dataset RedditHyperlinkTitle --model SignDyGFormer \
    --pattern "RAS-E.RASE-E.BTE-E.CNAS-E.TE"
# 与消融基线（全模块关闭）对比显著性:
python compute_stats.py --task sign --dataset RedditHyperlinkTitle --model SignDyGFormer \
    --pattern "RAS-E.RASE-E.BTE-E.CNAS-E.TE" --compare "RAS-D.RASE-D.BTE-D.CNAS-D.TE"
```

### 主表修复后重跑（RAE bug 连带，E-5 之后视时间执行）
> RAE 修复改变了完整模型行为：旧主表 `sign-ms.csv` / `linksign_ms.csv` 是用 **RAE 空操作**的旧代码跑出的（`np.append` 未赋值 bug），修订版主表必须用修复后代码（RAS/RASE/BTE/CNAS 全开）重跑，否则主表与消融/完整模型不一致。

步骤：
1. **影响评估（必做，快）**：BitcoinAlpha sign 任务 5 种子，与旧主表 `sign-ms.csv` 中 BitcoinAlpha（旧 AP≈0.9528）对比：
   ```bash
   python run_experiments.py -s sign -t main -m SignDyGFormer \
       -r BitcoinOTC RedditHyperlinkTitle RedditHyperlinkBody WikiVote -g 0 -e
   ```
   若指标变化 < 0.5%（经验阈值），可只重跑 E-5 已覆盖的 RedditTitle；否则执行全量重跑。
2. **全量重跑（视评估结果）**：5 数据集 × 2 任务 × 5 种子（**仅 SignDyGFormer**，基线模型不受 RAE 影响无需重跑）：
   ```bash
   python run_experiments.py -s sign -t main -m SignDyGFormer -g 0 -e
   python run_experiments.py -s linksign -t main -m SignDyGFormer -g 0 -e
   ```
   注：BitcoinAlpha/BitcoinOTC 很快，WikiVote/RedditBody@20000 中等，RedditTitle 已被 E-5 覆盖。
   重跑后主表数值以新结果为准，供 Agent A 更新表格。

### E-7 噪声鲁棒性（P6）—— 本轮（9 月）不执行，方案与模块已保留
> 决定：本轮不执行 E-7。`utils/noise.py` 可插拔模块已实现（可迁移 SEMBA 仓库），
> `--noise-ratio/--noise-seed/--noise-scope` 参数已接入，后续如需补充实验直接启用即可。

**已实现的能力**：
- `SignFlipNoise(noise_ratio, seed)`：`flip_mask` 基于【全局边顺序+seed】确定性生成；
- 本仓库侧 `--noise-ratio/--noise-seed/--noise-scope(train|all)`（结果文件名带 `.N{比例}{T|A}` 标记）；
- SEMBA 侧：`apply_temporal(train_data)` 翻转 `y`（y∈{0,1}，1=正 0=负 → `1-y`；与 msg 不重叠，只翻 y）。

**SEMBA 迁移用法**：
```python
from utils.noise import SignFlipNoise  # 把 utils/noise.py 拷到 SEMBA 仓库
data = dataset[0].to(device)
train_data, val_data, test_data = data.train_val_test_split(val_ratio=0.15, test_ratio=0.15)
SignFlipNoise(0.1, seed=0).apply_temporal(train_data)   # train-only 加噪
# 全数据加噪: dataset = SEMBADataset(..., pre_transform=SignFlipNoise(0.1, seed=0))
```

**本仓库执行**（数据集：BitcoinAlpha + WikiVote@20000；噪声级别 0.1/0.3；对比 SignDyG vs DyGFormer）：
```bash
# SignDyG 完整模型，train-only 噪声，0.1 / 0.3，5 种子
python run_experiments.py -s sign -t main -m SignDyGFormer \
    -r BitcoinOTC RedditHyperlinkTitle RedditHyperlinkBody WikiVote -g 0 -e --extra-noise 0.1   # 待封装
```
> 说明：当前 run_experiments 尚无噪声透传参数，若用 run_experiments 批量跑需先加 `--noise-*` 透传；
> 否则直接用训练脚本单条命令（含 `--noise-ratio 0.1 --noise-seed 0 --noise-scope train`）。
> 噪声结果与干净结果文件名不同（`.N10T` 等），不会互相覆盖。

## 2. 结果位置与交付

| 内容 | 路径 |
|---|---|
| 每个 run 的结果 JSON（含 E-1 效率 4 项） | `./saved_results/{LinkSign\|SignLinkPrediction}/{model}/{dataset}/{result_save_name}.json` |
| profiler 推理时间明细 | 同目录 `{...}-profiler.json` |
| run_experiments 运行日志 | `./expm-YYYY-MM-DD-logs/{task}/...log` |
| E-5 统计汇总 | `compute_stats.py` 输出（可 `--output x.csv`） |

## 3. 优化建议

- **E-1 并入 E-5**：E-5 的 sign 任务 seed42 结果 JSON 已含 `single run time / training time / inference time / peak memory / parameter count` 全部 4 项效率数据，E-1 无需单独跑。
- **RAE bug 已修复**（`np.append` 未赋值）：旧结果中 RASE=E/D 对比行无效，**旧主表数据在新代码下不可直接引用**，重跑前不要复用。
- 若 E-2 的 RedditTitle@20000 单次 run 远超预期，可评估降级为 tail 10000（文件已存在），但会与主表(20000)不一致，需先与 Agent A 确认。

## 4. 每完成一个实验立即记录

开始/结束时间、异常情况、结果文件路径 → 存入运行日志目录，供 Agent A 汇总。
