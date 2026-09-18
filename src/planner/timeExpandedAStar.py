"""Rolling time-expanded A-star comparison planner."""

from __future__ import annotations

import heapq
import math

from model.weatherMap import FREE, WeatherMapSnapshot

from .globalPlanner import PlanResult, TimedWaypoint

_MOVES = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0))


class TimeExpandedAStarPlanner:
    def __init__(
        self,
        *,
        horizon_steps: int = 30,
        step_seconds: float = 60.0,
        helicopter_speed_knots: float = 50.0,
        allow_wait: bool = True,
    ) -> None:
        self.horizon_steps = horizon_steps
        self.step_seconds = step_seconds
        self.helicopter_speed_knots = helicopter_speed_knots
        self.allow_wait = allow_wait

    @staticmethod
    def _grid_at(
        frames: tuple[WeatherMapSnapshot, ...], time_step: int
    ) -> tuple[tuple[int, ...], ...]:
        return frames[min(time_step, len(frames) - 1)].global_grid

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

        def cell(point: tuple[float, float]) -> tuple[int, int] | None:
            column = math.floor(point[0] / resolution)
            row = math.floor(point[1] / resolution)
            if 0 <= row < height and 0 <= column < width:
                return row, column
            return None

        start_cell = cell(start_nm)
        if start_cell is None:
            return PlanResult((), False, 0, math.inf, "start outside map")

        def target_at(step: int) -> tuple[int, int] | None:
            hours = step * self.step_seconds / 3600.0
            return cell(
                (
                    frigate_start_nm[0] + frigate_velocity_nm_per_hour[0] * hours,
                    frigate_start_nm[1] + frigate_velocity_nm_per_hour[1] * hours,
                )
            )

        start_state = (start_cell[0], start_cell[1], 0)
        frontier = [(0.0, 0.0, start_state)]
        came_from: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start_state: None}
        cost_so_far = {start_state: 0.0}
        expanded = 0
        goal_state: tuple[int, int, int] | None = None

        while frontier:
            _, current_cost, current = heapq.heappop(frontier)
            if current_cost != cost_so_far.get(current):
                continue
            row, column, time_step = current
            expanded += 1
            if target_at(time_step) == (row, column):
                goal_state = current
                break
            if time_step >= self.horizon_steps:
                continue

            candidates = list(_MOVES)
            if self.allow_wait:
                candidates.append((0, 0))
            for dr, dc in candidates:
                next_row, next_column = row + dr, column + dc
                if not (0 <= next_row < height and 0 <= next_column < width):
                    continue
                distance_nm = resolution * math.hypot(dr, dc)
                travel_steps = (
                    1
                    if distance_nm == 0
                    else max(
                        1,
                        math.ceil(
                            distance_nm / self.helicopter_speed_knots * 3600.0 / self.step_seconds
                        ),
                    )
                )
                next_time = time_step + travel_steps
                if next_time > self.horizon_steps:
                    continue
                grid = self._grid_at(weather_frames, next_time)
                if grid[next_row][next_column] != FREE:
                    continue
                next_state = (next_row, next_column, next_time)
                step_cost = travel_steps + (0.05 if (dr, dc) == (0, 0) else 0.0)
                new_cost = current_cost + step_cost
                if new_cost >= cost_so_far.get(next_state, math.inf):
                    continue
                cost_so_far[next_state] = new_cost
                came_from[next_state] = current
                target = target_at(next_time)
                heuristic = (
                    0.0
                    if target is None
                    else math.hypot(target[0] - next_row, target[1] - next_column)
                )
                heapq.heappush(frontier, (new_cost + heuristic, new_cost, next_state))

        if goal_state is None:
            return PlanResult((), False, expanded, math.inf, "no goal in horizon")
        states = []
        cursor: tuple[int, int, int] | None = goal_state
        while cursor is not None:
            states.append(cursor)
            cursor = came_from[cursor]
        states.reverse()
        waypoints = tuple(
            TimedWaypoint(
                (column + 0.5) * resolution,
                (row + 0.5) * resolution,
                time_step,
            )
            for row, column, time_step in states
        )
        return PlanResult(waypoints, True, expanded, cost_so_far[goal_state])


__all__ = ["TimeExpandedAStarPlanner"]
