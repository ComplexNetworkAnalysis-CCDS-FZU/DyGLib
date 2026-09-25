# -*- coding: utf-8 -*-
from pathlib import Path

out = Path("tools/queue/reorder_gridfirst_20260925.txt").read_text(encoding="utf-8").splitlines()
checks = [
    (0, "num-neighbors 15 --common-neighbors-look-forward 1", "grid RT linksign NN15 LF1"),
    (24, "num-neighbors 100 --common-neighbors-look-forward 15", "grid RT linksign NN100 LF15"),
    (149, "train_link_sign_prediction", "grid last (sign)"),
    (150, "dataset-name RedditHyperlinkBody", "B5 RB first"),
    (154, "dataset-name BitcoinOTC", "B5 OTC last"),
    (155, "dataset-name RedditHyperlinkTitle", "signRT RT"),
    (156, "e2-self-recent 3", "E2 WV k3"),
    (185, "e2-self-recent 30", "E2 OTC k30"),
]
ok_all = True
for i, sub, label in checks:
    ok = sub in out[i]
    ok_all &= ok
    print(f"{i + 1:3d} [{label}] -> {'OK' if ok else 'MISS'}")
assert ok_all, "边界校验失败"
print("全部边界校验通过；总行数", len(out))
