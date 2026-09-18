"""Predict future free rendezvous cells on the frigate trajectory."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Intercept:
    time_hours: float
    point_nm: tuple[float, float]


def predict_intercept(
    helicopter_nm: tuple[float, float],
    frigate_nm: tuple[float, float],
    frigate_velocity_nm_per_hour: tuple[float, float],
    helicopter_speed_knots: float,
) -> Intercept | None:
    """Solve the constant-speed moving-target interception equation."""
    if helicopter_speed_knots <= 0:
        raise ValueError("helicopter_speed_knots must be positive")
    rx = frigate_nm[0] - helicopter_nm[0]
    ry = frigate_nm[1] - helicopter_nm[1]
    vx, vy = frigate_velocity_nm_per_hour
    a = vx * vx + vy * vy - helicopter_speed_knots**2
    b = 2.0 * (rx * vx + ry * vy)
    c = rx * rx + ry * ry
    roots: list[float] = []
    if abs(a) < 1e-12:
        if abs(b) > 1e-12:
            roots.append(-c / b)
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0:
            square_root = math.sqrt(discriminant)
            roots.extend(((-b - square_root) / (2.0 * a), (-b + square_root) / (2.0 * a)))
    positive = [root for root in roots if root >= 0 and math.isfinite(root)]
    if not positive:
        return None
    time_hours = min(positive)
    return Intercept(
        time_hours,
        (
            frigate_nm[0] + vx * time_hours,
            frigate_nm[1] + vy * time_hours,
        ),
    )


def future_free_intercepts(
    frigate_nm: tuple[float, float],
    frigate_velocity_nm_per_hour: tuple[float, float],
    *,
    horizon_minutes: float,
    interval_seconds: float,
    is_free: Callable[[tuple[float, float], int], bool],
) -> tuple[Intercept, ...]:
    candidates: list[Intercept] = []
    step_count = math.floor(horizon_minutes * 60.0 / interval_seconds)
    for step in range(step_count + 1):
        time_hours = step * interval_seconds / 3600.0
        point = (
            frigate_nm[0] + frigate_velocity_nm_per_hour[0] * time_hours,
            frigate_nm[1] + frigate_velocity_nm_per_hour[1] * time_hours,
        )
        if is_free(point, step):
            candidates.append(Intercept(time_hours, point))
    return tuple(candidates)


__all__ = ["Intercept", "future_free_intercepts", "predict_intercept"]
