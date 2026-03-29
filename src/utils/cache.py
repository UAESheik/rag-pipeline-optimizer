from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


class SimpleCache:
    """基于文件系统的简单缓存，可按开关禁用。"""

    def __init__(self, cache_dir: Path, enabled: bool = False):
        self.cache_dir = cache_dir
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    @staticmethod
    def make_key(payload: Dict[str, Any]) -> str:
        """将输入负载序列化后生成稳定哈希键。"""
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Dict[str, Any] | None:
        """读取缓存，不存在或禁用时返回 None。"""
        if not self.enabled:
            return None
        p = self._path(key)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def set(self, key: str, value: Dict[str, Any]) -> None:
        """写入缓存；禁用时跳过。"""
        if not self.enabled:
            return
        self._path(key).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
