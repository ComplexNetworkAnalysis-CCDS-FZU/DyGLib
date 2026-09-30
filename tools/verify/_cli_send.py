"""CLI 信箱发送器（MCP 不可用时的契约通道）。

用法：
  python tools/verify/_cli_send.py <text_file> --to Paper --type 交付 --token TK [--ref ID ...]
"""
import subprocess
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

args = sys.argv[1:]
text_file = Path(args[0])
rest = args[1:]
to = None
typ = "FYI"
token = None
refs = []
i = 0
while i < len(rest):
    if rest[i] == "--to":
        to = rest[i + 1]; i += 2
    elif rest[i] == "--type":
        typ = rest[i + 1]; i += 2
    elif rest[i] == "--token":
        token = rest[i + 1]; i += 2
    elif rest[i] == "--ref":
        refs.append(rest[i + 1]); i += 2
    else:
        sys.exit(f"未知参数 {rest[i]}")

text = text_file.read_text(encoding="utf-8").strip()
cmd = [sys.executable, "-m", "mailbox.cli", "send", "--from", "Code", "--to", to,
       "--type", typ, "--text", text]
for r in refs:
    cmd += ["--ref", r]
if token:
    cmd += ["--token", token]
res = subprocess.run(cmd, cwd=r"D:\codes\agent-mailbox", capture_output=True, text=True,
                     encoding="utf-8", errors="replace")
print("RC =", res.returncode)
print(res.stdout)
if res.stderr:
    print("STDERR:", res.stderr)
