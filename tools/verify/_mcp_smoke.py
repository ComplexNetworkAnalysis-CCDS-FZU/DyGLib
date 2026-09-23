# -*- coding: utf-8 -*-
"""mcp_server.py stdio 冒烟：initialize + tools/list，验证信箱 MCP 壳可用。"""
import json
import subprocess
import sys
import time

PY = sys.executable
SRV = r"D:\codes\agent-mailbox\mcp_server.py"

p = subprocess.Popen(
    [PY, "-X", "utf8", SRV],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    encoding="utf-8",
)
reqs = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize",
     "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "diag", "version": "0"}}},
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
]
for m in reqs:
    p.stdin.write(json.dumps(m) + "\n")
    p.stdin.flush()
time.sleep(1.5)
p.stdin.close()
try:
    out, err = p.communicate(timeout=15)
except subprocess.TimeoutExpired:
    p.kill()
    out, err = p.communicate()
    print("!! 服务器未退出（超时 kill）")

print("=== STDOUT ===")
for line in out.splitlines()[:12]:
    print(line[:400])
print("=== STDERR ===")
print(err[:800])
print("=== RC:", p.returncode)
