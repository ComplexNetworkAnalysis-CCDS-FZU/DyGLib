# -*- coding: utf-8 -*-
"""校验恢复用的原尾部文件（432-617 原版）：B5(5) + signRT(1) + E2(30) + grid(150)。"""
from pathlib import Path

p = Path("tools/queue/_tail_432_617_20260925.txt")
lines = p.read_text(encoding="utf-8").splitlines()
print("总行数:", len(lines))
checks = [
    (0, "dataset-name RedditHyperlinkBody", "b5"),
    (4, "dataset-name BitcoinOTC", "b5"),
    (5, "no-module-balance-theory-encoder", "signRT"),
    (6, "e2-self-recent 3", "E2 WV k3"),
    (35, "e2-self-recent 30", "E2 OTC k30"),
    (36, "num-neighbors 15 --common-neighbors-look-forward 1", "grid RT linksign NN15 LF1"),
    (185, "num-neighbors 100 --common-neighbors-look-forward 15", "grid WV sign NN100 LF15"),
]
ok_all = True
for i, sub, label in checks:
    ok = sub in lines[i]
    ok_all &= ok
    print(f"{i + 1:3d} [{label}] -> {'OK' if ok else 'MISS'}")
assert len(lines) == 186 and ok_all, "恢复文件校验失败"
print("恢复文件校验通过")
