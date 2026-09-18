"""Build stacked global/local weather and kinematic observations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from model.weatherMap import WeatherMapSnapshot


def _flatten(rows: Sequence[Sequence[int]]) -> tuple[int, ...]:
    return tuple(cell for row in rows for cell in row)


def build_observation(
    history: Sequence[WeatherMapSnapshot],
    *,
    helicopter_nm: tuple[float, float],
    frigate_nm: tuple[float, float],
    helicopter_heading_deg: float,
    helicopter_speed_knots: float,
    frigate_velocity_nm_per_hour: tuple[float, float],
    elapsed_seconds: float,
    maximum_seconds: float,
    action_mask: tuple[bool, ...],
    guidance_global: tuple[tuple[float, ...], ...] | None = None,
    guidance_vector: tuple[float, ...] = (),
) -> dict[str, Any]:
    """Return a dependency-free observation with immutable numeric values."""
    if not history:
        raise ValueError("weather history cannot be empty")
    latest = history[-1]
    width, height = latest.area_size_nm
    return {
        "global_weather": tuple(_flatten(frame.global_grid) for frame in history),
        "global_shape": (
            len(latest.global_grid),
            len(latest.global_grid[0]),
        ),
        "local_weather": tuple(_flatten(frame.local_grid) for frame in history),
        "local_shape": (
            len(latest.local_grid),
            len(latest.local_grid[0]),
        ),
        "local_origin_nm": latest.local_origin_nm,
        "kinematics": (
            helicopter_nm[0] / width,
            helicopter_nm[1] / height,
            frigate_nm[0] / width,
            frigate_nm[1] / height,
            (frigate_nm[0] - helicopter_nm[0]) / width,
            (frigate_nm[1] - helicopter_nm[1]) / height,
            helicopter_heading_deg / 360.0,
            helicopter_speed_knots / 50.0,
            frigate_velocity_nm_per_hour[0] / 50.0,
            frigate_velocity_nm_per_hour[1] / 50.0,
            min(1.0, elapsed_seconds / maximum_seconds),
        ),
        "action_mask": action_mask,
        "guidance_global": (
            _flatten(guidance_global)
            if guidance_global is not None
            else tuple(0.0 for _ in range(len(latest.global_grid) * len(latest.global_grid[0])))
        ),
        "guidance_vector": guidance_vector,
        "map_size_nm": latest.area_size_nm,
        "weather_version": latest.version,
    }


__all__ = ["build_observation"]
