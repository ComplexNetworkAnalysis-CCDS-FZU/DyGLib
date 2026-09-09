# SignDyG 修订实验进度记录（NEUCOM-D-26-13975）

> **维护者**：Agent B（代码与实验）
> **用途**：所有 Agent（含 Agent A 论文撰写）通过本文件了解最新进度。
> **规则**：每次实验状态变化立即更新本文件，并通过 git 同步（本地 push → 服务器 pull）。
> 提交截止：2026-10-10。

## 全局状态（2026-09-06 更新）

| 项目 | 状态 | 说明 |
|---|---|---|
| 服务器 | ✅ 可用 | 2026-09-01 恢复访问 |
| 代码同步 | ✅ 完成 | 已推送 `sign-adoption` 分支至服务器裸仓库并 clone；`6f72c2c` 已同步 |
| 数据就绪 | ✅ 完成 | `server_setup.sh` 已执行，WikiVote tail20000 已生成 |
| 环境安装 | ✅ 完成 | conda env `gc`（torch 2.2.2） |
| **GPU 驱动** | ✅ **已修复（2026-09-06）** | DKMS `nvidia/595.84` 已装（内核 6.8.0-124/138）；`torch.cuda.is_available()=True`，设备 `NVIDIA GeForce RTX 2080 SUPER`；fedsa 免 sudo 可访问。详见 `docs/GPU_DRIVER_INSTALL.md`（实际装 595.84，非计划 590） |
| 耗时预期 | ✅ 可切 GPU 口径 | 自 2026-09-06 起新实验可按 GPU 估算；E-3 及此前日志仍为 CPU 耗时 |
| E-7 噪声 | ⛔ 本轮不做 | 模块 `utils/noise.py` 已实现保留 |

## 实验状态总览

| 实验 | 任务 | 数据集 | 状态 | 结果路径 / 备注 |
|---|---|---|---|---|
| E-1 效率 | sign | RedditTitle@20000 | ✅ 可提取 | E-5 sign seed42 已含 4 项效率数据，待汇总 |
| E-2 消融 | linksign | **全部 5 数据集** | ✅ 20/20 完成(GPU) ⚠️ 结论受阻 | 见「审计发现」：RAS/RAE 实际无效果 → base≡full；是否重设计/补 BTE-CNAS 组**待决策** |
| E-3 Patch | linksign | WikiVote@20000 + RedditBody@20000 | ✅ **8/8 完成（GPU）** | 结果已汇总至 `results/E-3_patch/E3_patch_summary.md`；结论：P=1 最优/持平，大 patch 有损（详见汇总） |

## E-3 结果摘要（2026-09-07，GPU 基座，详见 results/E-3_patch/E3_patch_summary.md）
- RedditBody：AUC P1=0.9610 → P7=0.9400 单调下降（-0.021），大 patch 明显有损
- WikiVote：各 P 几乎持平（AUC 差 ≤0.0008），不敏感
- 结论：默认 patch_size=1 有据可依（回应 R2#8 patch 超参），无需改模型
| E-4 时序 | linksign | WikiVote@20000 | ✅ 完成(GPU) | TD(λ=1.0) AUC=0.9596 < TE 0.9634 → **时间编码 TE 更优**（保留现状）；原始在 results/E-4_time_decay/raw/ |
| E-5 显著性 | sign + linksign | RedditTitle@20000 | ✅ 10/10 完成(GPU) | 双任务×5 种子（42,123,456,789,1024）；原始在服务器 saved_results/{LinkSign,SignLinkPrediction}/SignDyGFormer/RedditHyperlinkTitle/；待汇总显著性/效率/主表
| 主表重跑 | sign + linksign | 5 数据集 | ⬜ 待执行(GPU) | 先 BitcoinAlpha 影响评估；基线模型一并 GPU 重跑 |
| E-6 异配图 | — | — | ⛔ 本轮不做 | — |

**执行顺序（固定，不跳步）**：E-3(GPU重跑) → E-2(GPU) → E-4 → E-5 → 主表重跑

**设备基座（2026-09-06）**：最终进论文表格的数据**统一 GPU**（同 seed 跨设备不可比，CPU/GPU 随机流不同）；CPU 期日志/结果仅作存档与冒烟参考，不进论文。结果 JSON 现含 `device` 字段可核验。

## 审计发现（2026-09-09，subagent 只读核查 + 数据实测）

> **RAS 与 RAE 实际为（近）无效果模块** —— 影响 E-2 消融结论与论文消融表述，**需与导师/Agent A 定夺**（决策入口已发 HANDOFF）。

1. **RAS（repeat_aware）语义错位**：`utils/direct_neighbor_sampler.py` `common_neighbor_location`（:94-108）的 repeat 分支只**改写已有键**（要求 dst/src 出现在自己的历史邻居里 = 自环），从不把普通重复对插入锚点。实测：Bitcoin/Reddit **0 自环 → 0 触发**；WikiVote 5 条自环 → 14/20000 (0.07%) 边触发 → 与 E-2（4 数据集 base≡full、仅 WikiVote 微差）**定量吻合**。
2. **RAE（np.append 修复）仍是 no-op**：`models/NeighborInteractEncoder.py:293-297` 修复赋值真实存在，但 append 的 src/dst 或本就在交集中（padded 序列位置 0 = 自身），或凑不成配对 → on/off 输出**逐元素等价**（单测 identical=True）。parameter count 相同因 RAE 复用 BTE 的 `neighbor_sign_effect_layer`（共享层，无新增参数）。
3. **启动脚本参数链路正确**：run_experiments MODULE_GROUP→子进程 flag、args→sampler/model、`-r` 排除 / `-e` 5 种子均无误。
4. **结论**：E-2 的 base≡full 是**真实恒同**（非舍入巧合）。若论文消融声称 RAS/RAE 贡献，在 4/5 数据集不成立。
5. **连带影响**：主表「full 模型」实际等价 BTE+CNAS；**E-3/E-4/E-5（全模型）数据仅在 C5=方案②（不修 RAS/RAE，full:=BTE+CNAS）下为最终口径**；若 C5=方案①（修好 RAS/RAE），full 模型将改变 → E-3/E-4/E-5/主表**全部需重跑**（E-3 已被 A 写入论文 §4.5，需 A 知悉此条件性）。

## 🔴 追加重大发现（2026-09-09，subagent 深挖 BTE）：**pos0 标签泄漏**
- `models/SignDyGFormer.py:534`：padded 序列 pos0（自身 token）的 sign = **当前待预测边 (u,v) 的标签符号**；训练与**评估**都传真标签（`evaluate_models_utils.py:147` 等）。
- BTE 共同邻居计数中，重复边 (u,v) 会使配对 `(u@seq(u)pos0 × u@seq(v)历史)`、`(v@seq(u)历史 × v@seq(v)pos0)`，`suggest_sign = 标签 × 历史符号`（`NeighborInteractEncoder.py:169-181`）→ **测试时标签泄漏**（仅重复边，Bitcoin 重复率 ~40%）。
- **影响面**：所有 BTE 开启的运行（E-2 四行全部 BTE=True、E-3/E-4/E-5/主表、旧主表）→ 重复边上指标**虚高**；基线模型（无 BTE）不受影响 → **full 模型超基线的核心结论被泄漏系统性高估**。RAE no-op 依旧。
- **修复方向**（subagent 建议）：计数前剔除 pos0/自身 id（indirect 只收真第三方 w∉{u,v}）+ RAE 改为在 counterpart 位置按历史符号直写 [1,0]/[0,1]（不用标签）→ 修复后需全量重跑。
- **修复状态（2026-09-09）**：✅ 已实现（改动 1-4，见 `docs/FIX_PLAN_RAS_RAE_LEAK.md`）+ 本地验证全绿（泄漏消除/RAE 增量/RAS 触发 31/300）+ **GPU 验证完成**：BitcoinAlpha linksign full（修复后）AUC=0.9649 与修复前完全一致（6 项指标 4 位小数全同）→ 该数据集上泄漏移除/RAS+RAE 激活**无可测影响**（泄漏信息冗余，历史符号已由 edge 通道提供）；RAS/RAE 边际贡献待修复后 E-2 消融确认。全量重跑（E-2/E-3/E-4/E-5/主表）为最终口径。

## 待决策（阻塞主表重跑与 E-2 定稿）
- **RAS/RAE 去留（C5）**：语义定义见 `docs/DESIGN_RAS_RAE_FIX.md`。**用户 09-09 定案：以论文定义为准**（git 溯源跳过）；已向 Agent A 请求摘录论文中 RAS/RAE 准确定义（HANDOFF）。Agent A 倾向方案②（不再声称 RAS/RAE，消融改 BTE/CNAS）。主表重跑已暂缓；E-4/E-5 汇总先行交付。
- 建议先与导师确认（含机制 A/B 与是否新增通道）。

## 每次运行后需记录

对每个 run（或每个实验），记录：
- **开始/结束时间**、耗时
- **异常情况**（如有）
- **结果文件路径**（`saved_results/...`）与关键指标
- 更新上方"实验状态总览"表格

## 已确认决策（2026-09-05 更新，以 ADVISOR_DECISIONS.md 为准）

- 数据集设置**保持 tail 现状**：RedditTitle / RedditBody / WikiVote tail20000；BitcoinAlpha/BitcoinOTC 全量
- **E-2 消融 = 导师方案 4 组 × 全部 5 数据集**（CNAS+BTE 基座，RAS/RAE 解绑；link&sign）；配置已固化
- E-2 旧方案（基线全关→逐加）在 RedditTitle/WikiVote 已跑完，其中"全开"配置可复用，新跑用 `--module-idx 0 1 2` 跳过
- E-3/E-4 用 **link&sign** 任务；E-5 用 **sign + link&sign 双任务 × 5 种子**（满足审稿人 R2#8）
- **RAE bug 已修复**（`np.append` 未赋值）：旧主表（sign-ms.csv/linksign_ms.csv）用旧代码跑出，须用修复后代码重跑；先 BitcoinAlpha 影响评估（阈值 0.5%）再决定全量
- E-6（异配图）、E-7（噪声）：本轮不做（回复中作 future work）
- SEMBA 在独立仓库，不在此实现

## 运行日志与结果位置

| 内容 | 路径 |
|---|---|
| run_experiments 批量日志 | `expm-YYYY-MM-DD-logs/{任务}/` |
| 结果 JSON（含 E-1 效率 4 项） | `saved_results/{LinkSign\|SignLinkPrediction}/{model}/{dataset}/` |
| profiler 推理明细 | 同目录 `{...}-profiler.json` |
| 统计汇总（mean±std + p 值） | `dataset_analysis/compute_stats.py` 输出 |

## 最近更新记录

- **2026-09-01**：服务器恢复；代码推送至服务器；`server_setup.sh`（数据就绪脚本）+ 本进度文件建立；E-7 确定本轮不执行。
- **2026-08-20**：E-1~E-7 全部代码实现完成（时间衰减、效率测量、消融/patch 脚本、统计脚本、噪声模块、RAE bug 修复），本地 CPU 冒烟测试全部通过；`EXPERIMENT_PLAN.md` 运行计划定稿。
