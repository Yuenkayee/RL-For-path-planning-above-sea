"""JSON checkpoint persistence for dependency-free agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_checkpoint(agent: Any, path: str | Path, *, metadata: dict | None = None) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"agent": agent.state_dict(), "metadata": metadata or {}}
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return output


def load_checkpoint(agent: Any, path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    agent.load_state_dict(payload["agent"])
    return payload.get("metadata", {})


__all__ = ["load_checkpoint", "save_checkpoint"]
