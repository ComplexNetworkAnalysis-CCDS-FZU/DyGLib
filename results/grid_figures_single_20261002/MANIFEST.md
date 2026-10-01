# 网格热力图「单面板紧凑件」交付 MANIFEST（Code → Paper · 2026-10-02）

## 内容
- **12 组** = 3 数据集（RedditTitle / RedditBody / WikiRfA）× 2 任务（linksign / sign）× 2 指标（主指标 / AUC）
  - linksign 主指标 = `f1wt`；sign 主指标 = `f1mac`；另一组 = `auc`
- **每组 2 档尺寸 × 2 格式（pdf/png）** ⇒ 48 文件：
  - **无后缀 = 按最终尺寸设计**（面板宽 **168 pt**、注释字号 **6.8 pt**、刻度 5.5 pt、轴名 6.0 pt）
    ⇒ 以 `0.32\textwidth`(≈154 pt) 置入时接近 1:1，最终字号 ≈6.5–6.8 pt（可读）；
  - **`_natural` = 自然尺寸件**（面板宽 **400 pt**、注释 7.5 pt）⇒ 若按自然尺寸(≈0.8\textwidth)置入则用这档。

## 与 `8b2f` 要求的对照 / 一处算术说明（需你确认口径）
1. **已去红框**（不再标注"当前点"）+ 面板内不再出现 dataset/task 重复标题与图例（保留 NN/LF 轴名，标题仅简短 metric 名）。
2. **单面板独立文件** ✓，命名 `fig_grid_{task}_{ds}_{metric}.{pdf,png}`。
3. ⚠️ **算术说明**：`8b2f §一.3` 同时给出"页面宽 ≈380–420 pt + 单元格字号 ≈7–8 pt"与"缩到 `0.32\textwidth`(≈154 pt)"。
   这两条互斥：按 400 pt 设计再缩到 154 pt 时 7.5 pt 会变成 **≈2.9 pt**（与上次 16 in/18.9 pt 缩到 ≈2.5 pt 同理）。
   故交**两档**：默认件按"**最终尺寸**"设计（168 pt / 6.8 pt ⇒ 缩到 0.32\textwidth 后仍 ≈6.5 pt）；
   `_natural` 件按你字面给的 400 pt/7.5 pt 设计（供自然尺寸置入）。**请按你的实际置入宽度选档**。
4. 小件为给 4 位小数留宽度，**去掉了色标条**（每格数值已标注；如需色标可用 `_natural` 件或我另加窄色标）。

## 代际证据（可复现）
- 重跑 `tools/fig/gen_grid_heatmaps.py` ⇒ `figures/fig_grid_heatmap.csv` sha256 = **BF35908F1ED32E29BBD13F375C5A42746C20C7C576B5EC407ECEBC1F0ACE2058**（与 10-01 交付**逐位相同**）
- 数据源：`results/grid_new/raw/{linksign,sign}/{ds}/SignDyGFormer_seed42.NN-*.LF-*.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json`（new generation 2026-09-25；**单种子 seed42**；每组 25 格全覆盖）

## Bitcoin 覆盖（回答 `8b2f §二`）
- **同代际（乃至任何代际）都没有 Bitcoin 网格**：`grid_new/raw` 仅 RT/RB/WV；全库检索无 Bitcoin 网格文件。
- ⇒ 按你的分支②：**3 datasets**，每图一行 3 个；图注写"同代际网格覆盖 3 个数据集；Bitcoin 两集未参与本轮网格审计"。

## 文件 sha256（前 16 位）
- b7fcb046507381a4  fig_grid_linksign_RedditHyperlinkBody_auc.pdf  (9422 B)
- 45c0d5e1190eaec2  fig_grid_linksign_RedditHyperlinkBody_auc.png  (69752 B)
- f91ac8e8c5702103  fig_grid_linksign_RedditHyperlinkBody_auc_natural.pdf  (11587 B)
- 4ea7a3bcccaddd38  fig_grid_linksign_RedditHyperlinkBody_auc_natural.png  (107043 B)
- 6cec3e1c5fdb5df8  fig_grid_linksign_RedditHyperlinkBody_f1wt.pdf  (11165 B)
- 7bb399ff26034413  fig_grid_linksign_RedditHyperlinkBody_f1wt.png  (73890 B)
- 54d5da534b2a2cdf  fig_grid_linksign_RedditHyperlinkBody_f1wt_natural.pdf  (13386 B)
- 7a69e7ff23bed5b9  fig_grid_linksign_RedditHyperlinkBody_f1wt_natural.png  (118881 B)
- 6d64815975d37559  fig_grid_linksign_RedditHyperlinkTitle_auc.pdf  (9481 B)
- 1c9bb4d91e56da97  fig_grid_linksign_RedditHyperlinkTitle_auc.png  (74480 B)
- c38e634ecf0b9984  fig_grid_linksign_RedditHyperlinkTitle_auc_natural.pdf  (11694 B)
- f566a10da0a1d08d  fig_grid_linksign_RedditHyperlinkTitle_auc_natural.png  (118937 B)
- 4ce14cd6002e0f87  fig_grid_linksign_RedditHyperlinkTitle_f1wt.pdf  (11160 B)
- 753bcaff7f1de73c  fig_grid_linksign_RedditHyperlinkTitle_f1wt.png  (70489 B)
- 882e7bccb2b73bdd  fig_grid_linksign_RedditHyperlinkTitle_f1wt_natural.pdf  (13386 B)
- 6e9e06af177e7373  fig_grid_linksign_RedditHyperlinkTitle_f1wt_natural.png  (121754 B)
- 0763100eaffc6bec  fig_grid_linksign_WikiVote_auc.pdf  (9400 B)
- 26930495b50ff6f8  fig_grid_linksign_WikiVote_auc.png  (74986 B)
- aa8011c7cc937042  fig_grid_linksign_WikiVote_auc_natural.pdf  (11634 B)
- 3240b9a5805fa554  fig_grid_linksign_WikiVote_auc_natural.png  (114814 B)
- 8884a0fb646f1a00  fig_grid_linksign_WikiVote_f1wt.pdf  (11116 B)
- 63980a84a1cf8516  fig_grid_linksign_WikiVote_f1wt.png  (72574 B)
- c2118339742d236b  fig_grid_linksign_WikiVote_f1wt_natural.pdf  (13371 B)
- d3996c38e5c75914  fig_grid_linksign_WikiVote_f1wt_natural.png  (120417 B)
- 3edd41565af9d763  fig_grid_sign_RedditHyperlinkBody_auc.pdf  (9428 B)
- cc0a533bdeea5cec  fig_grid_sign_RedditHyperlinkBody_auc.png  (78475 B)
- 57ddd36c2ddc03ca  fig_grid_sign_RedditHyperlinkBody_auc_natural.pdf  (11625 B)
- 8eb08f04191aafc5  fig_grid_sign_RedditHyperlinkBody_auc_natural.png  (111853 B)
- 3983faec99c7b1fa  fig_grid_sign_RedditHyperlinkBody_f1mac.pdf  (12334 B)
- 20a94e14c110b42a  fig_grid_sign_RedditHyperlinkBody_f1mac.png  (53059 B)
- 72aadf25796be226  fig_grid_sign_RedditHyperlinkBody_f1mac_natural.pdf  (14494 B)
- 8152b3519abfe28d  fig_grid_sign_RedditHyperlinkBody_f1mac_natural.png  (86721 B)
- 6c06ca6a03800a65  fig_grid_sign_RedditHyperlinkTitle_auc.pdf  (9473 B)
- 09237aae927ae508  fig_grid_sign_RedditHyperlinkTitle_auc.png  (75761 B)
- 901619de00e658e7  fig_grid_sign_RedditHyperlinkTitle_auc_natural.pdf  (11665 B)
- 90a490290326ad9f  fig_grid_sign_RedditHyperlinkTitle_auc_natural.png  (109913 B)
- cea725f6eedda558  fig_grid_sign_RedditHyperlinkTitle_f1mac.pdf  (12437 B)
- 87e8bca8f3b82339  fig_grid_sign_RedditHyperlinkTitle_f1mac.png  (53447 B)
- 21cb0edd60389b70  fig_grid_sign_RedditHyperlinkTitle_f1mac_natural.pdf  (14576 B)
- 7c427ae11e30b186  fig_grid_sign_RedditHyperlinkTitle_f1mac_natural.png  (81236 B)
- 8dac885ea7c9ec62  fig_grid_sign_WikiVote_auc.pdf  (9443 B)
- afbd19d2f948d8fe  fig_grid_sign_WikiVote_auc.png  (66964 B)
- 603c8117e3752370  fig_grid_sign_WikiVote_auc_natural.pdf  (11689 B)
- 71ed383be9e33cdd  fig_grid_sign_WikiVote_auc_natural.png  (109725 B)
- aa88057346e716f8  fig_grid_sign_WikiVote_f1mac.pdf  (12641 B)
- af351928385dfc53  fig_grid_sign_WikiVote_f1mac.png  (74818 B)
- 729bce585443449d  fig_grid_sign_WikiVote_f1mac_natural.pdf  (14888 B)
- def79c332abfa850  fig_grid_sign_WikiVote_f1mac_natural.png  (113500 B)
