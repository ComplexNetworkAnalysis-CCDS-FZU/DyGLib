D6 附录底表：配置与来源（定稿值 = 泄漏修复后 5 种子复核批；早期候选来源不可回溯为泄漏时代专属）

| 任务 | 数据集 | NN | LF | m | P | tail | 来源文件（seed42） | sha256[:16] | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| linksign | WikiVote | 15 | 10 | 0 | 1 | 20000 | `results/e1a_tailfill/raw_base/linksign/WikiVote/SignDyGFormer_seed42.NN-15.LF-10.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json` | `1eac7c2e4a0f22fa` | 修复后主表批定稿（main-c）；E1c 探索列 m=8–11（未进主表） |
| linksign | RedditHyperlinkTitle | 60 | 1 | 0 | 1 | 20000 | `results/e1a_tailfill/raw_base/linksign/RedditHyperlinkTitle/SignDyGFormer_seed42.NN-60.LF-1.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json` | `7516ebf85ae69476` | 修复后主表批定稿（main-b） |
| linksign | RedditHyperlinkBody | 80 | 3 | 0 | 1 | 20000 | `results/e1a_tailfill/raw_base/linksign/RedditHyperlinkBody/SignDyGFormer_seed42.NN-80.LF-3.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json` | `a77a4536eebbb014` | 修复后主表批定稿（main-b）；E1c 探索列 m=80（未进主表） |
| linksign | BitcoinAlpha | 40 | 15 | 0 | 1 | — | `results/e1a_tailfill/raw_base/linksign/BitcoinAlpha/SignDyGFormer_seed42.NN-40.LF-15.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json` | `d2ca08a92f1bba3c` | 修复后主表批定稿（main-e） |
| linksign | BitcoinOTC | 80 | 5 | 0 | 1 | — | `results/e1a_tailfill/raw_base/linksign/BitcoinOTC/SignDyGFormer_seed42.NN-80.LF-5.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json` | `51331020bdc6c553` | 修复后主表批定稿（main-e） |
| sign | WikiVote | 40 | 15 | 0 | 1 | 20000 | `results/sign_valthr/raw/WikiVote/SignDyGFormer_seed42.NN-40.LF-15.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json` | `dceec0ed23005425` | 修复后 5 种子复核批定稿（nh5/main） |
| sign | RedditHyperlinkTitle | 100 | 1 | 0 | 1 | 20000 | `results/sign_valthr/raw/RedditHyperlinkTitle/SignDyGFormer_seed42.NN-100.LF-1.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json` | `f56fecd1bedc3690` | ⓓ 候选 100/3 5 种子 n.s.（09-14）→ 未换装；定稿 = 修复后复核批 |
| sign | RedditHyperlinkBody | 60 | 1 | 0 | 1 | 20000 | `results/sign_valthr/raw/RedditHyperlinkBody/SignDyGFormer_seed42.NN-60.LF-1.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json` | `fe64c5c5ca2228b4` | ⓓ 候选 40/1 5 种子 n.s.（09-14）→ 未换装；定稿 = 修复后复核批 |
| sign | BitcoinAlpha | 40 | 15 | 0 | 1 | — | `results/sign_valthr/raw/BitcoinAlpha/SignDyGFormer_seed42.NN-40.LF-15.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json` | `4ceb6c67c3492593` | 修复后 5 种子复核批定稿（main-d） |
| sign | BitcoinOTC | 40 | 15 | 0 | 1 | — | `results/sign_valthr/raw/BitcoinOTC/SignDyGFormer_seed42.NN-40.LF-15.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json` | `0d42effed1d462f7` | 09-14 用户拍板换装 60/10→40/15（0.8775±0.0054 vs 0.8696±0.0048，配对 Δ+0.0079、t=3.16）；09-17 同步进 TASK 表 |

注：全部 10 行与 `run_experiments.py::TASK_DATASET_BEST_PARAMS` 一致；P=1 为默认（E-3 扫描无一致趋势）；m=0 主表默认。
