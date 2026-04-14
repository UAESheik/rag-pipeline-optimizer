from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


def read_json(path: Path) -> Dict:
    """读取 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict]:
    """读取 JSONL 文件为字典列表。"""
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> List[Dict]:
    """读取 CSV 文件。"""
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    """写入 CSV 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_json(path: Path, obj: Dict) -> None:
    """写入 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
