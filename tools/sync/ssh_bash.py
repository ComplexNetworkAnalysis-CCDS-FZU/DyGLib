# -*- coding: utf-8 -*-
"""ssh_bash.py — 把本地 bash 脚本经 ssh 在服务器执行（stdin 流）。

规避 PowerShell→ssh 管道的 CRLF/BOM/转义坑：subprocess 直送 stdin；
并在远端先 `tr -d '\r'` 再交给 bash（Windows ssh 客户端 stdin 会夹带 CR，
bash 对 CR 敏感；Python 脚本因容忍 CRLF 不受影响）。

用法（仓库根）：
    python tools/sync/ssh_bash.py <local_script.sh>
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SSH_TARGET = "fedsa@172.17.173.102"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    script = path.read_text(encoding="utf-8-sig").replace("\r", "")
    r = subprocess.run(
        ["ssh", SSH_TARGET, "tr -d '\\r' | bash -s"],
        input=script,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    print(r.stdout, end="")
    if r.stderr.strip():
        print("--- stderr ---")
        print(r.stderr, end="")
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
