"""Common data structures for rolling-horizon global planners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from model.weatherMap import WeatherMapSnapshot


@dataclass(frozen=True)
class TimedWaypoint:
    x_nm: float
    y_nm: float
    time_step: int


@dataclass(frozen=True)
class PlanResult:
    waypoints: tuple[TimedWaypoint, ...]
    reached_goal: bool
    expanded_states: int
    cost: float
    reason: str = ""


class GlobalPlanner(Protocol):
    def plan(
        self,
        weather_frames: tuple[WeatherMapSnapshot, ...],
        start_nm: tuple[float, float],
        frigate_start_nm: tuple[float, float],
        frigate_velocity_nm_per_hour: tuple[float, float],
    ) -> PlanResult: ...


__all__ = ["GlobalPlanner", "PlanResult", "TimedWaypoint"]
