# 网格热力图交付 MANIFEST（Code → Paper · 2026-10-01）

来源：生成器 tools/fig/gen_grid_heatmaps.py（docstring: new generation, 2026-09-25）
源数据：results/grid_new/raw/{task}/{ds}/SignDyGFormer_seed42.NN-*.LF-*.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json（150 格）

## 代际确认（回答 Paper 的问题）
- **是同代际（new generation / 重跑版）**，不是 6 月旧版参数图。
- 证据：交付前重跑一次生成器 → figures/fig_grid_heatmap.csv 的 sha256 **前后逐位相同**（BF35908F…2058）⇒ 图与当前 grid_new 归档**逐位可复现**一致。
- 旧版参数图（fig/exp/figs/*-img.pdf）未被本次流程使用。

## 文件 sha256（前 16 位）
- bf35908f1ed32e29  fig_grid_heatmap.csv  (7269 B)
- e1d7b51841c3dec3  fig_grid_linksign_RedditHyperlinkBody.pdf  (23163 B)
- f24fd2384d58201a  fig_grid_linksign_RedditHyperlinkBody.png  (383726 B)
- fd344148653461f0  fig_grid_linksign_RedditHyperlinkTitle.pdf  (22834 B)
- da31dca928e7d499  fig_grid_linksign_RedditHyperlinkTitle.png  (406119 B)
- f2d21877fd61378b  fig_grid_linksign_WikiVote.pdf  (22955 B)
- 8f3ff848c9a37da4  fig_grid_linksign_WikiVote.png  (398916 B)
- c83b182c1d5a3cfc  fig_grid_sign_RedditHyperlinkBody.pdf  (22807 B)
- 9ca887f7dbb0528b  fig_grid_sign_RedditHyperlinkBody.png  (335388 B)
- 50e6a5338a590849  fig_grid_sign_RedditHyperlinkTitle.pdf  (22450 B)
- 594073da2147ea69  fig_grid_sign_RedditHyperlinkTitle.png  (332077 B)
- 656321e571d3ee9a  fig_grid_sign_WikiVote.pdf  (22754 B)
- c796d12bbf0b544b  fig_grid_sign_WikiVote.png  (379344 B)

## 用法
- 6 张图 = {linksign,sign} × {RedditTitle,RedditBody,WikiRfA}，双面板（主指标 | AUC）；PNG 供排版，PDF 供矢量。
- fig_grid_heatmap.csv = 150 格长表（task,dataset,NN,LF,主指标,auc），机读。
