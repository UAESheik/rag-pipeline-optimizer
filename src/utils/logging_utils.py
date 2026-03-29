from __future__ import annotations

from pathlib import Path


def append_jsonl(path: Path, line: str) -> None:
    """向 jsonl 文件追加一行文本。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
