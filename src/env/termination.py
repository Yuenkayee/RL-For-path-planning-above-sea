"""Success, collision, boundary and timeout termination rules."""

from __future__ import annotations

import math
from typing import Protocol

from model.weatherMap import FREE


class _WeatherLike(Protocol):
    area_size_nm: tuple[float, float]

    def weather_at(self, x_nm: float, y_nm: float) -> int: ...


def inside_map(point_nm: tuple[float, float], area_size_nm: tuple[float, float]) -> bool:
    return 0.0 <= point_nm[0] < area_size_nm[0] and 0.0 <= point_nm[1] < area_size_nm[1]


def same_cell(
    first_nm: tuple[float, float],
    second_nm: tuple[float, float],
    resolution_nm: float,
) -> bool:
    return (
        math.floor(first_nm[0] / resolution_nm),
        math.floor(first_nm[1] / resolution_nm),
    ) == (
        math.floor(second_nm[0] / resolution_nm),
        math.floor(second_nm[1] / resolution_nm),
    )


def successful_rendezvous(
    weather_map: _WeatherLike,
    helicopter_nm: tuple[float, float],
    frigate_nm: tuple[float, float],
    *,
    resolution_nm: float,
) -> bool:
    if not same_cell(helicopter_nm, frigate_nm, resolution_nm):
        return False
    return weather_map.weather_at(*helicopter_nm) == FREE


__all__ = ["inside_map", "same_cell", "successful_rendezvous"]
