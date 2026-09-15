"""edit_remote_tasks.py — 通过 ssh + python 安全编辑服务器上的队列文件 tools/queue/tasks.txt（零 sed/零转义）。

背景（2026-09-14 教训）：经 PowerShell → ssh 的 sed/正则转义可能被吃掉（`\\r` 曾被毁成 `r`，
毁掉三行任务）。本工具全程不走 shell 转义：读取 = ssh 内 python 直读；
回写 = 新内容 base64 后由服务器端 python 解码写盘；写入前自动备份（tasks.txt.bak-<时间戳>），
写入后回读校验逐行一致。

用法（仓库根目录）：
    python tools/queue/edit_remote_tasks.py --show 77,84
    python tools/queue/edit_remote_tasks.py --insert-after 82 --lines-file tools/queue/insert_s1_rerun.txt
"""
from __future__ import annotations

import argparse
import base64
import subprocess
import sys

SSH = "fedsa@172.17.173.102"
REMOTE = "cd ~/DyGLib && python -"


def ssh_py(code: str) -> str:
    res = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=15", SSH, REMOTE],
        input=code,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    if res.returncode != 0:
        sys.exit(f"[ERR] ssh 失败：{(res.stderr or '')[-500:]}")
    return res.stdout


def read_tasks() -> list:
    out = ssh_py(
        "import pathlib\n"
        "print(pathlib.Path('tools/queue/tasks.txt').read_text(encoding='utf-8'), end='')"
    )
    return out.rstrip("\n").splitlines()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--insert-after", type=int, default=0, help="在第 N 行后插入（1-based）")
    ap.add_argument("--lines-file", help="要插入的行（本地文件，逐行）")
    ap.add_argument("--show", help="如 77,84 —— 打印 [a,b] 行区间")
    args = ap.parse_args()

    lines = read_tasks()
    print(f"[remote] tasks.txt 当前 {len(lines)} 行")

    if args.show:
        a, b = (int(x) for x in args.show.split(","))
        for i in range(a - 1, min(b, len(lines))):
            print(f"{i + 1}: {lines[i]}")
        return

    if not args.insert_after or not args.lines_file:
        sys.exit("需要 --insert-after 与 --lines-file（或 --show）")

    with open(args.lines_file, encoding="utf-8") as f:
        new_lines = [l for l in f.read().splitlines() if l.strip()]
    n = args.insert_after
    if not (1 <= n <= len(lines)):
        sys.exit(f"[ERR] 插点 {n} 超出范围 1..{len(lines)}")

    merged = lines[:n] + new_lines + lines[n:]
    b64 = base64.b64encode(("\n".join(merged) + "\n").encode("utf-8")).decode("ascii")

    code = (
        "import base64, datetime, pathlib, shutil\n"
        "p = pathlib.Path('tools/queue/tasks.txt')\n"
        "bak = p.parent / (p.name + '.bak-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))\n"
        "shutil.copy(p, bak)\n"
        "p.write_bytes(base64.b64decode('__B64__'))\n"
        "print('BAK', bak.name)\n"
    ).replace("__B64__", b64)
    out = ssh_py(code)
    print("[write]", out.strip())

    actual = read_tasks()
    if actual == merged:
        print(f"[ok] 回读校验通过：{len(actual)} 行（旧 {len(lines)} + 新 {len(new_lines)}）")
        for i in range(n, n + len(new_lines)):
            print(f"{i + 1}: {actual[i]}")
    else:
        first = next(
            (i for i, (a, b) in enumerate(zip(actual, merged)) if a != b),
            min(len(actual), len(merged)),
        )
        print(f"[ERR] 回读与预期不一致（首个差异行 {first + 1}）；请检查备份文件后重试")
        print("  actual:", actual[first] if first < len(actual) else "<EOF>")
        print("  expect:", merged[first] if first < len(merged) else "<EOF>")
        sys.exit(1)


if __name__ == "__main__":
    main()
