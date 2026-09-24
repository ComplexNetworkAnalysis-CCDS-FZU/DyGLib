#!/bin/bash
cd /home/fedsa/DyGLib
echo "########## WV noBTE param.json ##########"
python3 -m json.tool "saved_models/SignLinkPrediction/SignDyGFormer/WikiVote/SignDyGFormer_seed42/SignDyGFormer_seed42.NN-Best.LF-Best.RAS-E.RASE-E.BTE-D.CNAS-E.P1.TE.param.json" | head -80
echo ""
echo "########## RB G1 param.json ##########"
python3 -m json.tool "saved_models/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody/SignDyGFormer_seed42/SignDyGFormer_seed42.NN-80.LF-3.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.G1.param.json" | head -80
