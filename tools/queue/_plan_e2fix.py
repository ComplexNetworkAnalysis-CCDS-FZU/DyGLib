"""构造 E2 修复 + 网格去重的新 519-641 队列块（一次性替换）。

输入：
  tools/queue/_dump_519_647_20260926.txt   （服务器 tasks.txt 519-647 行）
  tools/queue/e2_stage1_20260925.txt       （E2 段一 30 行源文件）
处理：
  1) E2 补行 12 条：e2 源文件中 BA/OTC 的 12 行，去掉 ` --tail-num 20000`（BA/OTC 无 tail 数据，规范=全量）
  2) 网格去重：519-641 区间内删除 LF30 重复格 25 行（6 个 best-NN 组，每组 LF-1/3/5/10/15）
     - linksign(SignLinkPrediction, train_sign_link_3class)：RT NN-60、RB NN-80、WV NN-15
     - sign(LinkSign, train_link_sign)：RT NN-100、RB NN-60、WV NN-40
输出：
  tools/queue/e2_fix12_grid_trim_20260926.txt （12 补行 + 98 保留网格行 = 110 行）
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Q = ROOT / "tools" / "queue"

SCRIPT2TASK = {
    "train_sign_link_3class_prediction.py": "linksign",   # -> SignLinkPrediction/
    "train_link_sign_prediction.py": "sign",              # -> LinkSign/
}
LF30_SET = {
    ("linksign", "RedditHyperlinkTitle", 60),
    ("linksign", "RedditHyperlinkBody", 80),
    ("linksign", "WikiVote", 15),
    ("sign", "RedditHyperlinkTitle", 100),
    ("sign", "RedditHyperlinkBody", 60),
    ("sign", "WikiVote", 40),
}
LFS = {1, 3, 5, 10, 15}


def parse(line):
    py = re.search(r"(train_[a-z0-9_]+\.py)", line)
    ds = re.search(r"--dataset-name ([^ ]+)", line)
    nn = re.search(r"--num-neighbors (\d+)", line)
    lf = re.search(r"--common-neighbors-look-forward (\d+)", line)
    return (
        py.group(1) if py else None,
        ds.group(1) if ds else None,
        int(nn.group(1)) if nn else None,
        int(lf.group(1)) if lf else None,
    )


dump = (Q / "_dump_519_647_20260926.txt").read_text(encoding="utf-8").splitlines()
print(f"dump 行数 = {len(dump)}（服务器行 519-647）")

deleted, keep, tail6 = [], [], []
for i, line in enumerate(dump):
    rowno = 519 + i
    if rowno > 641:
        tail6.append((rowno, line))
        continue
    py, ds, nn, lf = parse(line)
    task = SCRIPT2TASK.get(py)
    hit = task and ds and nn is not None and lf is not None and (task, ds, nn) in LF30_SET and lf in LFS
    if hit:
        deleted.append((rowno, task, ds, nn, lf))
    else:
        keep.append((rowno, line))
        tag = f"{task}/{ds}/NN{nn}/LF{lf}" if task else "??"
        if task is None:
            print(f"  [warn] 519-641 区间未识别行: {rowno}: {line[:100]}")

print(f"\n删除（LF30 重复格）= {len(deleted)} 行：")
for rowno, task, ds, nn, lf in deleted:
    print(f"  del #{rowno}: {task:8s} {ds:22s} NN-{nn} LF-{lf}")
print(f"保留网格行 = {len(keep)}（应为 98）；尾部原样 642-647 = {len(tail6)}（应为 6）")

e2 = (Q / "e2_stage1_20260925.txt").read_text(encoding="utf-8").splitlines()
fix = [l for l in e2 if re.search(r"--dataset-name (BitcoinAlpha|BitcoinOTC)", l)]
assert len(fix) == 12, f"BA/OTC E2 行数 {len(fix)} != 12"
fix2 = []
for l in fix:
    l2 = l.replace(" --tail-num 20000 --e2-self-recent", " --e2-self-recent")
    assert "--tail-num" not in l2, l2
    fix2.append(l2)
print(f"\nE2 补行 = {len(fix2)} 条（已去 --tail-num）：")
for l in fix2:
    m = re.search(r"(train_[a-z0-9_]+\.py) --dataset-name ([^ ]+).*--e2-self-recent (\d+)", l)
    print(f"  fix: {SCRIPT2TASK.get(m.group(1), '?'):8s} {m.group(2):14s} k={m.group(3)}")

newblock = fix2 + [l for _, l in keep]
out = Q / "e2_fix12_grid_trim_20260926.txt"
with open(out, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(newblock) + "\n")
print(f"\n[ok] 写出 {out.relative_to(ROOT)}（{len(newblock)} 行 = 12 补行 + {len(keep)} 网格行）")
print(f"[chk] 替换后文件总行数应为 518 + {len(newblock)} + 6 = {518 + len(newblock) + 6}")
