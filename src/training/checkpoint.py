"""PyTorch checkpoint persistence for agents and optimizer state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(agent: Any, path: str | Path, *, metadata: dict | None = None) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"agent": agent.state_dict(), "metadata": metadata or {}}
    torch.save(payload, output)
    return output


def load_checkpoint(agent: Any, path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    agent.load_state_dict(payload["agent"])
    return payload.get("metadata", {})


__all__ = ["load_checkpoint", "save_checkpoint"]
