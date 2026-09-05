# SignDyG 修订协作契约（NEUCOM-D-26-13975）

双 Agent 协作修订（A=论文，B=代码/实验）。会话工作区 = 本仓库时，本文件自动加载。
分工（互不越界）：**Agent A** = 论文正文/理论/回复信（工作区 `D:\Sign_DygFormer`）；**Agent B** = 代码/实验/数据/服务器（本仓库）。

## 权威层级
- ★ 所有决策以 `docs/ADVISOR_DECISIONS.md` 为准，冲突时以此为准。
- `docs/REVIEWER_COMMENTS.md` 审稿归档；`docs/EXPERIMENT_PLAN.md` 实验计划；`docs/PROGRESS.md` 进度与结果。

## 开工必读 / 收工必写
1. 会话开始：读 `docs/HANDOFF.md`（信箱）→ `docs/ADVISOR_DECISIONS.md` → 相关 `docs/PROGRESS.md`。
2. 任务结束：更新 `docs/PROGRESS.md`；给对方的交付/请求登记 `docs/HANDOFF.md`（【日期|发出方|状态⬜→🔄→✅】，接收方处理后改状态）。
3. Agent A 在其他工作区：以绝对路径 `D:\codes\DyGLib\docs\` 直接读写（同机即时互见），其工作区根已有 `copilot-instructions.md` 指针。

## 修改纪律
- 涉及代码/实验/数据的修改须经用户批准后执行。
- B 的实验结果以「表格 + 结论」写入 `docs/PROGRESS.md`，供 A 直接引用到论文。
