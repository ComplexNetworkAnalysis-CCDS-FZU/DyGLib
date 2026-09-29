#!/bin/bash
cd ~/DyGLib || exit 1
echo '=== G2 param.json thresholds (FX rows) ==='
for spec in "RedditHyperlinkTitle:15:3" "RedditHyperlinkBody:60:1"; do
  ds=${spec%%:*}; rest=${spec#*:}; nn=${rest%%:*}; lf=${rest##*:}
  for s in 42 123 456 789 1024; do
    F="saved_models/SignLinkPrediction/SignDyGFormer/${ds}/SignDyGFormer_seed${s}/SignDyGFormer_seed${s}.NN-${nn}.LF-${lf}.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.G2.param.json"
    echo -n "$ds NN-${nn}.LF-${lf} seed${s}: "
    cat "$F" 2>/dev/null | tr '\n' ' ' || echo "MISSING"
    echo
  done
done
echo
echo '=== candidate val specs (.G2) ==='
cat > /tmp/cand_specs.txt <<'EOF'
linksign,RedditHyperlinkTitle,15,3,42,cand
linksign,RedditHyperlinkTitle,15,3,123,cand
linksign,RedditHyperlinkTitle,15,3,456,cand
linksign,RedditHyperlinkTitle,15,3,789,cand
linksign,RedditHyperlinkTitle,15,3,1024,cand
linksign,RedditHyperlinkBody,60,1,42,cand
linksign,RedditHyperlinkBody,60,1,123,cand
linksign,RedditHyperlinkBody,60,1,456,cand
linksign,RedditHyperlinkBody,60,1,789,cand
linksign,RedditHyperlinkBody,60,1,1024,cand
sign,RedditHyperlinkTitle,60,3,42,cand
sign,RedditHyperlinkTitle,60,3,123,cand
sign,RedditHyperlinkTitle,60,3,456,cand
sign,RedditHyperlinkTitle,60,3,789,cand
sign,RedditHyperlinkTitle,60,3,1024,cand
sign,RedditHyperlinkBody,40,1,42,cand
sign,RedditHyperlinkBody,40,1,123,cand
sign,RedditHyperlinkBody,40,1,456,cand
sign,RedditHyperlinkBody,40,1,789,cand
sign,RedditHyperlinkBody,40,1,1024,cand
sign,WikiVote,15,10,42,cand
sign,WikiVote,15,10,123,cand
sign,WikiVote,15,10,456,cand
sign,WikiVote,15,10,789,cand
sign,WikiVote,15,10,1024,cand
EOF
cat > /tmp/ctrl_specs.txt <<'EOF'
linksign,RedditHyperlinkTitle,60,1,42,ctrl
linksign,RedditHyperlinkTitle,60,1,123,ctrl
linksign,RedditHyperlinkTitle,60,1,456,ctrl
linksign,RedditHyperlinkTitle,60,1,789,ctrl
linksign,RedditHyperlinkTitle,60,1,1024,ctrl
linksign,RedditHyperlinkBody,80,3,42,ctrl
linksign,RedditHyperlinkBody,80,3,123,ctrl
linksign,RedditHyperlinkBody,80,3,456,ctrl
linksign,RedditHyperlinkBody,80,3,789,ctrl
linksign,RedditHyperlinkBody,80,3,1024,ctrl
sign,RedditHyperlinkTitle,100,1,42,ctrl
sign,RedditHyperlinkTitle,100,1,123,ctrl
sign,RedditHyperlinkTitle,100,1,456,ctrl
sign,RedditHyperlinkTitle,100,1,789,ctrl
sign,RedditHyperlinkTitle,100,1,1024,ctrl
sign,RedditHyperlinkBody,60,1,42,ctrl
sign,RedditHyperlinkBody,60,1,123,ctrl
sign,RedditHyperlinkBody,60,1,456,ctrl
sign,RedditHyperlinkBody,60,1,789,ctrl
sign,RedditHyperlinkBody,60,1,1024,ctrl
sign,WikiVote,40,15,42,ctrl
sign,WikiVote,40,15,123,ctrl
sign,WikiVote,40,15,456,ctrl
sign,WikiVote,40,15,789,ctrl
sign,WikiVote,40,15,1024,ctrl
EOF
echo '=== candidate val (5 seeds, .G2 newest-only) ==='
python3 tools/verify/extract_val_from_logs.py --spec-file /tmp/cand_specs.txt --suffix .G2 --newest-only | tee /tmp/cand_val.jsonl
echo '=== control val (5 seeds, newest-only) ==='
python3 tools/verify/extract_val_from_logs.py --spec-file /tmp/ctrl_specs.txt --newest-only | tee /tmp/ctrl_val.jsonl
