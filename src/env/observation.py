"""Build stacked global/local weather and kinematic observations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from model.weatherMap import WeatherMapSnapshot


def _array(rows: Sequence[Sequence[int | float]], dtype: np.dtype) -> NDArray:
    return np.asarray(rows, dtype=dtype)


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
    """Return NumPy arrays suitable for Gymnasium and PyTorch."""
    if not history:
        raise ValueError("weather history cannot be empty")
    latest = history[-1]
    width, height = latest.area_size_nm
    global_weather = np.stack([_array(frame.global_grid, np.uint8) for frame in history])
    local_weather = np.stack([_array(frame.local_grid, np.uint8) for frame in history])
    return {
        "global_weather": global_weather,
        "local_weather": local_weather,
        "kinematics": np.asarray(
            (
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
            dtype=np.float32,
        ),
        "action_mask": np.asarray(action_mask, dtype=np.int8),
        "guidance_global": (
            _array(guidance_global, np.float32)
            if guidance_global is not None
            else np.zeros(global_weather.shape[-2:], dtype=np.float32)
        ),
        "guidance_vector": np.asarray(guidance_vector, dtype=np.float32),
    }


__all__ = ["build_observation"]
