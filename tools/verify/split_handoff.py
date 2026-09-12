# -*- coding: utf-8 -*-
"""一次性迁移（2026-09-12）：docs/HANDOFF.md 单文件信箱 -> docs/handoff/outbox-*.md「一人一箱」。

用法（在仓库根目录运行）：
    python tools/verify/split_handoff.py

规则：
- 按节标题（收件人）+ 行内「发出方」列路由到对应发件箱（唯一写入者=箱主）；
- 条目内容逐字保留，仅把「发出方」列改为「收件人」列（收件人取自节标题）；
- 非 2026 数据行（表头/分隔/占位）跳过并计数。
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SRC = Path("docs/HANDOFF.md")
OUT_DIR = Path("docs/handoff")

# 节标题关键词（按先后顺序匹配）-> (收件人, {发出方: 箱名})
SECTIONS = [
    ("给 Paper · Code", "Paper/Code", {"Baseline": "baseline"}),
    ("给 Paper", "Paper", {"Code": "code", "B": "code"}),
    ("给 Code", "Code", {"Paper": "paper", "Perf": "perf"}),
    ("给 Baseline", "Baseline", {"Code": "code"}),
    ("给 Perf", "Perf", {"Code": "code"}),
]

OWNER_TITLE = {
    "code": "Code（原 B；历史条目中署名 B）",
    "paper": "Paper（原 A）",
    "baseline": "Baseline（原 C）",
    "perf": "Perf（2026-09-11 加入）",
}
OWNER_NAME = {"code": "Code", "paper": "Paper", "baseline": "Baseline", "perf": "Perf"}
RECIPIENT_ORDER = ["Paper", "Baseline", "Perf", "Code", "Paper/Code"]
EXPECTED = {"code": 15, "paper": 2, "baseline": 0, "perf": 7}


def file_header(owner: str) -> str:
    name = OWNER_NAME[owner]
    return (
        f"# 📤 发件箱 · {OWNER_TITLE[owner]}\n\n"
        f"> **唯一写入者：`{name}`**；其他 Agent 只读（请勿编辑本文件）。\n"
        f"> 本箱只放 `{name}` → 其他 Agent 的消息（收件人见第 2 列）。\n"
        f"> 收信：去其余三个发件箱找「收件人={name}」的行；处理完**在自己箱内加一条回执行**"
        f"（收件人=原发件人、状态 ✅、一句话结论），勿编辑对方文件。\n"
        f"> 入口/速览：`docs/HANDOFF.md` ｜ 历史归档：`docs/HANDOFF_ARCHIVE.md` ｜ "
        f"2026-09-12 拆箱（背景：单文件多写者曾致 2 次互相覆盖）。\n"
    )


def main() -> int:
    text = SRC.read_text(encoding="utf-8")
    rows: dict[str, list[tuple[str, str]]] = {k: [] for k in OWNER_NAME}
    cur = None
    skipped = 0
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("## ➡️ "):
            cur = None
            for key, recipient, senders in SECTIONS:
                if key in s:
                    cur = (recipient, senders)
                    break
            continue
        if cur is None or not s.startswith("|"):
            continue
        raw = s.split("|")
        if len(raw) < 6:
            skipped += 1
            continue
        cells = [c.strip() for c in raw[1:-1]]
        date, sender, status = cells[0], cells[1], cells[2]
        content = "|".join(cells[3:]).strip()
        if not date.startswith("2026"):
            skipped += 1
            continue
        recipient, senders = cur
        owner = senders.get(sender)
        if owner is None:
            print(f"[FAIL] 未映射的发出方 {sender!r}: {s[:80]}")
            return 1
        rows[owner].append((recipient, f"| {date} | {recipient} | {status} | {content} |"))

    if sum(len(v) for v in rows.values()) == 0:
        print("[FAIL] 未解析到任何条目——请检查源文件格式")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for owner in ("code", "paper", "baseline", "perf"):
        items = rows[owner]
        parts = [file_header(owner)]
        for recipient in RECIPIENT_ORDER:
            group = [r for (rc, r) in items if rc == recipient]
            if not group:
                continue
            parts.append(f"\n## ➡️ 给 {recipient}\n\n")
            parts.append("| 日期 | 收件人 | 状态 | 内容 |\n|---|---|---|---|\n")
            parts.append("\n".join(group) + "\n")
        if owner == "baseline" and not items:
            parts.append(
                "\n## ➡️ 给 Paper · Code\n\n| 日期 | 收件人 | 状态 | 内容 |\n|---|---|---|---|\n"
                "| — | — | — | （暂无活跃条目；历史 2 条见 `docs/HANDOFF_ARCHIVE.md`） |\n"
            )
        out = OUT_DIR / f"outbox-{owner}.md"
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            f.write("".join(parts))
        n = len(items)
        flag = "OK" if n == EXPECTED[owner] else "WARN"
        print(f"[{flag}] {out}  {n} 条（预期 {EXPECTED[owner]}）")

    print(f"[INFO] 跳过非数据行：{skipped}（预期 11：表头×5 + 分隔×5 + 占位×1）")
    print(f"[INFO] 迁移条目合计：{sum(len(v) for v in rows.values())}（预期 24）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
