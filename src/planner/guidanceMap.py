"""Rasterize global routes, corridors and timed subgoals for RL policies."""

from __future__ import annotations

import math

from .globalPlanner import TimedWaypoint


def rasterize_guidance(
    waypoints: tuple[TimedWaypoint, ...],
    *,
    shape: tuple[int, int],
    resolution_nm: float,
    corridor_radius_cells: int = 1,
) -> tuple[tuple[float, ...], ...]:
    """Return a normalized route-time channel for a global observation."""
    height, width = shape
    values = [[0.0] * width for _ in range(height)]
    denominator = max(1, len(waypoints) - 1)
    for index, waypoint in enumerate(waypoints):
        column = math.floor(waypoint.x_nm / resolution_nm)
        row = math.floor(waypoint.y_nm / resolution_nm)
        intensity = 1.0 - 0.5 * index / denominator
        for dy in range(-corridor_radius_cells, corridor_radius_cells + 1):
            for dx in range(-corridor_radius_cells, corridor_radius_cells + 1):
                rr, cc = row + dy, column + dx
                if 0 <= rr < height and 0 <= cc < width:
                    values[rr][cc] = max(values[rr][cc], intensity)
    return tuple(tuple(row) for row in values)


__all__ = ["rasterize_guidance"]
