# -*- coding: utf-8 -*-
"""构建尾部（462 起）：E2(30) + 网格150 + B5(5) + signRT(1) = 186 行（替换现 462-647）。

当前 462-647 = 网格(150) + B5(5) + signRT(1) + E2(30)；目标 = E2 + 网格 + B5 + signRT。
素材：tools/queue/_tail_432_617_20260925.txt（B5 idx0-4 / signRT idx5 / E2 idx6-35 / grid idx36-185）
输出：tools/queue/tail_e2_after_lf30_20260926.txt
"""
from pathlib import Path

q = Path("tools/queue")
src = q.joinpath("_tail_432_617_20260925.txt").read_text(encoding="utf-8").splitlines()
assert len(src) == 186
b5, signrt, e2, grid = src[0:5], src[5:6], src[6:36], src[36:186]

new_tail = e2 + grid + b5 + signrt
assert len(new_tail) == 186

out = q.joinpath("tail_e2_after_lf30_20260926.txt")
out.write_text("\n".join(new_tail) + "\n", encoding="utf-8")

checks = [
    (0, "e2-self-recent 3", "E2 WV k3"),
    (29, "e2-self-recent 30", "E2 OTC k30"),
    (30, "num-neighbors 15 --common-neighbors-look-forward 1", "grid RT NN15 LF1"),
    (179, "num-neighbors 100 --common-neighbors-look-forward 15", "grid last WV sign"),
    (180, "module-bte-b5", "B5 RB"),
    (184, "module-bte-b5", "B5 OTC"),
    (185, "no-module-balance-theory-encoder", "signRT"),
]
ok_all = True
for i, sub, label in checks:
    ok = sub in new_tail[i]
    ok_all &= ok
    print(f"{i + 1:3d} [{label}] -> {'OK' if ok else 'MISS'}")
assert ok_all, "边界校验失败"
print(f"总行数 {len(new_tail)}；写出 {out}")
