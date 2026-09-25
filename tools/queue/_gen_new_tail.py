# -*- coding: utf-8 -*-
"""构建目标尾部（432 起）：LF30(30) + 网格150 + B5(5) + signRT(1) + E2(30) = 216 行。

素材：
  - tools/queue/lf_sweep_30_20260925.txt（LF30）
  - tools/queue/_tail_432_617_20260925.txt（原尾部：B5 idx0-4 / signRT idx5 / E2 idx6-35 / grid idx36-185）
输出：tools/queue/tail_lf30_grid_b5_rt_e2_20260926.txt
"""
from pathlib import Path

q = Path("tools/queue")
lf30 = q.joinpath("lf_sweep_30_20260925.txt").read_text(encoding="utf-8").splitlines()
src = q.joinpath("_tail_432_617_20260925.txt").read_text(encoding="utf-8").splitlines()
assert len(lf30) == 30 and len(src) == 186
b5, signrt, e2, grid = src[0:5], src[5:6], src[6:36], src[36:186]
assert len(b5) == 5 and len(signrt) == 1 and len(e2) == 30 and len(grid) == 150

new_tail = lf30 + grid + b5 + signrt + e2
assert len(new_tail) == 216

out = q.joinpath("tail_lf30_grid_b5_rt_e2_20260926.txt")
out.write_text("\n".join(new_tail) + "\n", encoding="utf-8")

checks = [
    (0, "num-neighbors 60 --common-neighbors-look-forward 1", "LF30 RT linksign"),
    (29, "num-neighbors 40 --common-neighbors-look-forward 15", "LF30 WV sign"),
    (30, "num-neighbors 15 --common-neighbors-look-forward 1", "grid RT linksign NN15LF1"),
    (179, "num-neighbors 100 --common-neighbors-look-forward 15", "grid last WV sign"),
    (180, "module-bte-b5", "B5 RB"),
    (184, "module-bte-b5", "B5 OTC"),
    (185, "no-module-balance-theory-encoder", "signRT"),
    (186, "e2-self-recent 3", "E2 WV k3"),
    (215, "e2-self-recent 30", "E2 OTC k30"),
]
ok_all = True
for i, sub, label in checks:
    ok = sub in new_tail[i]
    ok_all &= ok
    print(f"{i + 1:3d} [{label}] -> {'OK' if ok else 'MISS'}")
assert ok_all, "边界校验失败"
print(f"总行数 {len(new_tail)}；写出 {out}")
