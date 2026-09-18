"""Validated environment configuration loading."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .reward import RewardWeights

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_CONFIG_PATH = REPOSITORY_ROOT / "config" / "envConfig.json"


@dataclass(frozen=True)
class EnvironmentConfig:
    control_step_seconds: float = 6.0
    weather_step_seconds: float = 60.0
    maximum_episode_minutes: float = 120.0
    heading_count: int = 16
    helicopter_speed_knots: tuple[float, float] = (0.0, 50.0)
    frigate_speed_knots: float = 30.0
    frigate_heading_deg: float = 225.0
    success_grid_resolution_nm: float = 0.1
    storm_safety_margin_nm: float = 0.1
    weather_history_frames: int = 6
    reward: RewardWeights = RewardWeights()

    def __post_init__(self) -> None:
        positive = (
            self.control_step_seconds,
            self.weather_step_seconds,
            self.maximum_episode_minutes,
            self.frigate_speed_knots,
            self.success_grid_resolution_nm,
        )
        if any(not math.isfinite(item) or item <= 0 for item in positive):
            raise ValueError("time, speed and resolution values must be positive")
        if self.heading_count < 4:
            raise ValueError("heading_count must be at least four")
        if len(self.helicopter_speed_knots) != 2:
            raise ValueError("helicopter_speed_knots must contain wait and cruise speeds")
        if self.helicopter_speed_knots[0] < 0 or self.helicopter_speed_knots[1] <= 0:
            raise ValueError("invalid helicopter speed modes")
        ratio = self.weather_step_seconds / self.control_step_seconds
        if not math.isclose(ratio, round(ratio), abs_tol=1e-9):
            raise ValueError("weather step must be an integer multiple of control step")
        if self.weather_history_frames < 1:
            raise ValueError("weather_history_frames must be positive")

    @property
    def action_count(self) -> int:
        return self.heading_count + 1

    @property
    def wait_speed_knots(self) -> float:
        return self.helicopter_speed_knots[0]

    @property
    def flight_speed_knots(self) -> float:
        return self.helicopter_speed_knots[1]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> EnvironmentConfig:
        data = dict(values)
        if "helicopter_speed_knots" in data:
            speeds = data["helicopter_speed_knots"]
            if not isinstance(speeds, (list, tuple)) or len(speeds) != 2:
                raise ValueError("helicopter_speed_knots must contain two values")
            data["helicopter_speed_knots"] = (float(speeds[0]), float(speeds[1]))
        if "reward" in data:
            data["reward"] = RewardWeights.from_mapping(data["reward"])
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_ENV_CONFIG_PATH) -> EnvironmentConfig:
        with Path(path).expanduser().resolve().open(encoding="utf-8") as stream:
            return cls.from_mapping(json.load(stream))


__all__ = ["DEFAULT_ENV_CONFIG_PATH", "EnvironmentConfig"]
