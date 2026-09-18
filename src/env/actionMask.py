"""Mask unsafe wait and heading actions using swept-path collision checks."""

from __future__ import annotations

import math
from typing import Protocol

from model.weatherMap import FREE


class _WeatherLike(Protocol):
    area_size_nm: tuple[float, float]
    local_resolution_nm: float

    def weather_at(self, x_nm: float, y_nm: float) -> int: ...


def heading_for_action(action: int, heading_count: int) -> float | None:
    """Return heading clockwise from north; action zero means wait."""
    if action == 0:
        return None
    if not 1 <= action <= heading_count:
        raise ValueError(f"action must be in [0, {heading_count}]")
    return (action - 1) * 360.0 / heading_count


def endpoint(
    position_nm: tuple[float, float],
    heading_deg: float,
    speed_knots: float,
    duration_seconds: float,
) -> tuple[float, float]:
    distance_nm = speed_knots * duration_seconds / 3600.0
    angle = math.radians(heading_deg)
    return (
        position_nm[0] + distance_nm * math.sin(angle),
        position_nm[1] + distance_nm * math.cos(angle),
    )


def segment_is_clear(
    weather_map: _WeatherLike,
    start_nm: tuple[float, float],
    end_nm: tuple[float, float],
    *,
    sample_spacing_nm: float | None = None,
) -> bool:
    """Check map bounds and every sampled point of a swept segment."""
    spacing = sample_spacing_nm or weather_map.local_resolution_nm / 2.0
    distance = math.dist(start_nm, end_nm)
    sample_count = max(1, math.ceil(distance / spacing))
    width, height = weather_map.area_size_nm
    for index in range(sample_count + 1):
        fraction = index / sample_count
        x = start_nm[0] + (end_nm[0] - start_nm[0]) * fraction
        y = start_nm[1] + (end_nm[1] - start_nm[1]) * fraction
        if not (0.0 <= x < width and 0.0 <= y < height):
            return False
        if weather_map.weather_at(x, y) != FREE:
            return False
    return True


def build_action_mask(
    weather_map: _WeatherLike,
    position_nm: tuple[float, float],
    *,
    heading_count: int,
    flight_speed_knots: float,
    wait_speed_knots: float,
    duration_seconds: float,
    current_heading_deg: float = 0.0,
) -> tuple[bool, ...]:
    """Return one validity flag for wait plus each discrete flight heading."""
    mask: list[bool] = []
    for action in range(heading_count + 1):
        heading = heading_for_action(action, heading_count)
        if heading is None:
            destination = endpoint(
                position_nm, current_heading_deg, wait_speed_knots, duration_seconds
            )
        else:
            destination = endpoint(position_nm, heading, flight_speed_knots, duration_seconds)
        mask.append(segment_is_clear(weather_map, position_nm, destination))
    return tuple(mask)


__all__ = [
    "build_action_mask",
    "endpoint",
    "heading_for_action",
    "segment_is_clear",
]
