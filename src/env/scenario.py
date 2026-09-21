"""Deterministic episode geometry and dynamic-route difficulty measurements."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from model.weatherMap import FREE, WeatherMapSnapshot
from planner.interceptPredictor import Intercept, predict_intercept


@dataclass(frozen=True)
class RouteConflictSummary:
    """Thunderstorm encounters along the no-weather nominal intercept trajectory."""

    regions: int
    occupied_samples: int
    total_samples: int

    @property
    def occupied_fraction(self) -> float:
        return self.occupied_samples / self.total_samples if self.total_samples else 0.0


def sample_frigate_initial_position(
    rng: random.Random,
    *,
    map_size_nm: tuple[float, float],
    helicopter_nm: tuple[float, float],
    heading_deg: float,
    frigate_speed_knots: float,
    helicopter_speed_knots: float,
    route_minutes: float,
    boundary_margin_nm: float,
    minimum_distance_nm: float,
    maximum_attempts: int = 1_000,
) -> tuple[tuple[float, float], Intercept]:
    """Sample a catchable frigate start whose route stays in bounds for the duration."""
    route_hours = route_minutes / 60.0
    heading_rad = math.radians(heading_deg)
    displacement = (
        frigate_speed_knots * route_hours * math.sin(heading_rad),
        frigate_speed_knots * route_hours * math.cos(heading_rad),
    )
    lower_x = boundary_margin_nm + max(0.0, -displacement[0])
    upper_x = map_size_nm[0] - boundary_margin_nm - max(0.0, displacement[0])
    lower_y = boundary_margin_nm + max(0.0, -displacement[1])
    upper_y = map_size_nm[1] - boundary_margin_nm - max(0.0, displacement[1])
    if lower_x > upper_x or lower_y > upper_y:
        raise ValueError(
            "map is too small for the frigate to remain in bounds for the required route"
        )

    velocity = (
        frigate_speed_knots * math.sin(heading_rad),
        frigate_speed_knots * math.cos(heading_rad),
    )
    for _ in range(maximum_attempts):
        candidate = (rng.uniform(lower_x, upper_x), rng.uniform(lower_y, upper_y))
        if math.dist(helicopter_nm, candidate) < minimum_distance_nm:
            continue
        intercept = predict_intercept(
            helicopter_nm,
            candidate,
            velocity,
            helicopter_speed_knots,
        )
        if intercept is None or intercept.time_hours > route_hours:
            continue
        if not (
            boundary_margin_nm <= intercept.point_nm[0] < map_size_nm[0] - boundary_margin_nm
            and boundary_margin_nm
            <= intercept.point_nm[1]
            < map_size_nm[1] - boundary_margin_nm
        ):
            continue
        return candidate, intercept
    raise RuntimeError(
        "could not sample a catchable frigate start; reduce the minimum distance or margin"
    )


def count_nominal_route_conflicts(
    weather_frames: tuple[WeatherMapSnapshot, ...],
    *,
    start_nm: tuple[float, float],
    intercept: Intercept,
    forecast_step_minutes: float,
    sample_interval_minutes: float = 0.25,
) -> RouteConflictSummary:
    """Count separate storm intervals on a time-synchronised direct intercept route."""
    if not weather_frames:
        raise ValueError("weather_frames cannot be empty")
    if forecast_step_minutes <= 0.0 or sample_interval_minutes <= 0.0:
        raise ValueError("route-conflict sampling intervals must be positive")
    duration_minutes = intercept.time_hours * 60.0
    sample_count = max(2, math.ceil(duration_minutes / sample_interval_minutes) + 1)
    regions = 0
    occupied_samples = 0
    previously_occupied = False
    for index in range(sample_count):
        fraction = index / (sample_count - 1)
        elapsed_minutes = duration_minutes * fraction
        frame_index = min(
            len(weather_frames) - 1,
            round(elapsed_minutes / forecast_step_minutes),
        )
        frame = weather_frames[frame_index]
        x_nm = start_nm[0] + (intercept.point_nm[0] - start_nm[0]) * fraction
        y_nm = start_nm[1] + (intercept.point_nm[1] - start_nm[1]) * fraction
        column = math.floor(x_nm / frame.global_resolution_nm)
        row = math.floor(y_nm / frame.global_resolution_nm)
        occupied = bool(
            0 <= row < len(frame.global_grid)
            and 0 <= column < len(frame.global_grid[0])
            and frame.global_grid[row][column] != FREE
        )
        occupied_samples += occupied
        if occupied and not previously_occupied:
            regions += 1
        previously_occupied = occupied
    return RouteConflictSummary(regions, occupied_samples, sample_count)


__all__ = [
    "RouteConflictSummary",
    "count_nominal_route_conflicts",
    "sample_frigate_initial_position",
]
