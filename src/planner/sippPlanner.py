"""Safe Interval Path Planning baseline with explicit wait actions."""

from __future__ import annotations

import heapq
import math

from model.weatherMap import FREE, WeatherMapSnapshot

from .globalPlanner import PlanResult, TimedWaypoint

_NEIGHBORS = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))


def safe_intervals(
    weather_frames: tuple[WeatherMapSnapshot, ...],
    *,
    horizon_steps: int,
) -> dict[tuple[int, int], tuple[tuple[int, int], ...]]:
    """Build inclusive safe intervals for every low-resolution cell."""
    if not weather_frames:
        raise ValueError("weather_frames cannot be empty")
    height = len(weather_frames[0].global_grid)
    width = len(weather_frames[0].global_grid[0])
    result: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    for row in range(height):
        for column in range(width):
            intervals: list[tuple[int, int]] = []
            start: int | None = None
            for step in range(horizon_steps + 1):
                grid = weather_frames[min(step, len(weather_frames) - 1)].global_grid
                free = grid[row][column] == FREE
                if free and start is None:
                    start = step
                if not free and start is not None:
                    intervals.append((start, step - 1))
                    start = None
            if start is not None:
                intervals.append((start, horizon_steps))
            result[(row, column)] = tuple(intervals)
    return result


class SIPPPlanner:
    def __init__(
        self,
        *,
        horizon_steps: int = 30,
        step_seconds: float = 60.0,
        helicopter_speed_knots: float = 50.0,
    ) -> None:
        self.horizon_steps = horizon_steps
        self.step_seconds = step_seconds
        self.helicopter_speed_knots = helicopter_speed_knots

    def plan(
        self,
        weather_frames: tuple[WeatherMapSnapshot, ...],
        start_nm: tuple[float, float],
        frigate_start_nm: tuple[float, float],
        frigate_velocity_nm_per_hour: tuple[float, float],
    ) -> PlanResult:
        if not weather_frames:
            raise ValueError("weather_frames cannot be empty")
        frame = weather_frames[0]
        resolution = frame.global_resolution_nm
        height, width = len(frame.global_grid), len(frame.global_grid[0])
        intervals = safe_intervals(weather_frames, horizon_steps=self.horizon_steps)

        def to_cell(point: tuple[float, float]) -> tuple[int, int] | None:
            row = math.floor(point[1] / resolution)
            column = math.floor(point[0] / resolution)
            return (row, column) if 0 <= row < height and 0 <= column < width else None

        start_cell = to_cell(start_nm)
        if start_cell is None:
            return PlanResult((), False, 0, math.inf, "start outside map")
        start_interval_index = next(
            (
                index
                for index, interval in enumerate(intervals[start_cell])
                if interval[0] <= 0 <= interval[1]
            ),
            None,
        )
        if start_interval_index is None:
            return PlanResult((), False, 0, math.inf, "start cell unsafe")

        def target_at(step: int) -> tuple[int, int] | None:
            hours = step * self.step_seconds / 3600.0
            return to_cell(
                (
                    frigate_start_nm[0] + frigate_velocity_nm_per_hour[0] * hours,
                    frigate_start_nm[1] + frigate_velocity_nm_per_hour[1] * hours,
                )
            )

        start_node = (start_cell[0], start_cell[1], start_interval_index)
        arrival = {start_node: 0}
        parent: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start_node: None}
        frontier = [(0.0, 0, start_node)]
        expanded = 0
        goal_node: tuple[int, int, int] | None = None
        goal_arrival: int | None = None

        while frontier:
            _, current_time, node = heapq.heappop(frontier)
            if current_time != arrival.get(node):
                continue
            if goal_arrival is not None and current_time >= goal_arrival:
                continue
            row, column, interval_index = node
            expanded += 1
            current_interval = intervals[(row, column)][interval_index]
            matching_time = next(
                (
                    candidate
                    for candidate in range(current_time, current_interval[1] + 1)
                    if target_at(candidate) == (row, column)
                ),
                None,
            )
            if matching_time is not None:
                if goal_arrival is None or matching_time < goal_arrival:
                    goal_node = node
                    goal_arrival = matching_time
            for dr, dc in _NEIGHBORS:
                next_cell = row + dr, column + dc
                if not (0 <= next_cell[0] < height and 0 <= next_cell[1] < width):
                    continue
                distance_nm = resolution * math.hypot(dr, dc)
                travel_steps = max(
                    1,
                    math.ceil(
                        distance_nm / self.helicopter_speed_knots * 3600.0 / self.step_seconds
                    ),
                )
                for next_index, safe in enumerate(intervals[next_cell]):
                    candidate_time = max(current_time + travel_steps, safe[0])
                    departure_time = candidate_time - travel_steps
                    if candidate_time > safe[1] or departure_time > current_interval[1]:
                        continue
                    next_node = (next_cell[0], next_cell[1], next_index)
                    if candidate_time >= arrival.get(next_node, math.inf):
                        continue
                    arrival[next_node] = candidate_time
                    parent[next_node] = node
                    target = target_at(candidate_time)
                    heuristic = (
                        0.0
                        if target is None
                        else math.hypot(target[0] - next_cell[0], target[1] - next_cell[1])
                    )
                    heapq.heappush(
                        frontier,
                        (candidate_time + heuristic, candidate_time, next_node),
                    )
                    break

        if goal_node is None:
            return PlanResult((), False, expanded, math.inf, "no goal in horizon")
        nodes = []
        cursor: tuple[int, int, int] | None = goal_node
        while cursor is not None:
            nodes.append(cursor)
            cursor = parent[cursor]
        nodes.reverse()
        waypoints = tuple(
            TimedWaypoint(
                (column + 0.5) * resolution,
                (row + 0.5) * resolution,
                goal_arrival if node == goal_node and goal_arrival is not None else arrival[node],
            )
            for node in nodes
            for row, column, _ in (node,)
        )
        return PlanResult(
            waypoints,
            True,
            expanded,
            float(goal_arrival if goal_arrival is not None else arrival[goal_node]),
        )


__all__ = ["SIPPPlanner", "safe_intervals"]
