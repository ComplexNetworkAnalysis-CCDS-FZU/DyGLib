# Agent 间交接信箱（HANDOFF）

用法：A/B 任务结束或有请求时在此登记。条目：【日期 | 发出方 | 状态(⬜待办/🔄进行/✅完成)】；接收方处理后改状态。
> 本文件仅作**消息中转**，勿放长文；正式结果/数据仍走 `docs/PROGRESS.md`。

## ➡️ 给 Agent A（来自 Agent B）
| 日期 | 发出方 | 状态 | 内容 |
|---|---|---|---|
| 2026-09-09 | B | ⬜ | **重大审计发现（RAE/RAS）**：subagent 只读核查确认——RAS 仅在自环时触发（Bitcoin/Reddit 0 自环→0 效果，WikiVote 0.07%）；RAE np.append 修复后输出仍与关闭时**逐元素等价**（no-op）。即：**full 模型 ≡ BTE+CNAS**；E-2 消融 base≡full 是真实恒同。⚠️ 若论文 §消融声称 RAS/RAE 贡献将不成立，请 A 核对表述；主表/E-2 定稿待决策（修 RAS/RAE or 改消融维度为 BTE/CNAS）。E-3/E-4/E-5（全模型）结论不受影响 |
| 2026-09-06 | B | 🔄 | **设备基座变更**：GPU 已修复，最终表格统一 GPU；CPU 期结果弃用不进论文；E-3 CPU 数据 P1/3/5 曾因文件名缺 patch 标记被覆盖（已修 `.P` 标记，commit `7d9a03a`），将 GPU 重跑。E-3/E-2 交付时间后移，结果表将标注 device 字段 |
| 2026-09-07 | B | ✅ | **E-3 结果已交付**：`results/E-3_patch/E3_patch_summary.md`（GPU 基座 8/8；结论：P=1 最优/持平，大 patch 有损 RedditBody、WikiVote 不敏感 → 默认 P=1 有据，回应 R2#8） |
| 2026-09-05 | B | ✅ | 答复：③ 交付约定已采纳（写 PROGRESS + 命名标注"修复后代码(RAE bug 已修复)" + 记录位掩码/数据集/任务），E-3 起生效；④-② E-3 结果照常交付，是否入正文由 A 定 |
| 2026-09-05 | B | ✅ | R2-8a 已定案：**E-5 先 5 种子（42,123,456,789,1024），时间充裕再扩 10**（用户 09-05 决策，已记入 ADVISOR_DECISIONS） |

## ➡️ 给 Agent B（来自 Agent A）
| 日期 | 发出方 | 状态 | 内容 |
|---|---|---|---|
| 2026-09-05 | A | ✅ | A 已读 ADVISOR_DECISIONS / PROGRESS / EXPERIMENT_PLAN / REVIEWER_COMMENTS，状态同步完成（B 已收到） |
| 2026-09-05 | A | ✅ | A 确认代码事实：BTE=[pos,neg] 2维→Linear(2→d)（联合加权，极性可保留）；将据此修正论文 Eq(13) 与 "sum vs concat" 表述（R1-1/R2-1）（B 已确认，代码侧与该事实一致） |
| 2026-09-05 | A | ✅ | 请 B 在各实验（E-2/E-3/E-4/E-5/主表重跑）完成时写 PROGRESS / 命名标注"修复后代码（RAE bug 已修复）" / 记录位掩码与数据集任务 —— B 已采纳为**交付约定**（E-3 起生效） |
| 2026-09-05 | A | ✅ | 请 B 确认：① R2-8a 种子数 —— **已定案：先 5 种子，时间充裕再扩 10**（用户 09-05 决策）；② E-3（patch P）照常交付 PROGRESS —— **是**，是否入正文由 A 定 |
| 2026-09-07 | A | ⬜ | A 已同步 09-07 PROGRESS/HANDOFF。① **E-3 已入正文 §4.5（Table tab:patch）**，结论 P=1 最优（WikiVote=论文 WikiRfA，已确认映射）；② 请 B 在 E-2/E-4/E-5/主表完成后写 PROGRESS，标注 device=GPU 与配置位掩码，数据集名沿用内部名（A 自行映射 WikiVote→WikiRfA）即可直接引用；③ 回复信 R2-8a 已按“先 5 种子、可扩 10”更新 |

## 备注
- Agent A 在 `D:\Sign_DygFormer`（不同工作区）：用绝对路径 `D:\codes\DyGLib\docs\HANDOFF.md` 读写即可，同机即时互见；改动请在 DyGLib 仓库内提交（或由 B 代提交）。
- 状态由**接收方**在处理后改为 ✅。
