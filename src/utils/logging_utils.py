from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict


def log_external_event(
    component: str,
    action: str,
    status: str,
    detail: str = "",
    extra: Dict[str, Any] | None = None,
) -> None:
    """外部调用事件：运行时直接打印到终端，不写文件。"""
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "component": component,
        "action": action,
        "status": status,
        "detail": detail,
        "extra": extra or {},
    }
    print(f"[external] {json.dumps(payload, ensure_ascii=False)}")
