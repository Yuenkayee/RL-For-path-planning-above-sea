"""Shared JSON configuration helper for training entry points."""

from __future__ import annotations

import json
from pathlib import Path


def load_algorithm_config(path: str | Path) -> dict:
    with Path(path).expanduser().resolve().open(encoding="utf-8") as stream:
        values = json.load(stream)
    values.pop("algorithm", None)
    values.pop("note", None)
    return values


__all__ = ["load_algorithm_config"]
