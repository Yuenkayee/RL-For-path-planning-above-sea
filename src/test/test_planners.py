from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.weatherMap import FREE, THUNDERSTORM, WeatherMap
from planner.interceptPredictor import predict_intercept
from planner.sippPlanner import SIPPPlanner, safe_intervals
from planner.timeExpandedAStar import TimeExpandedAStarPlanner


class PlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.weather_map = WeatherMap(
            area_size_nm=(10.0, 10.0),
            global_resolution_nm=1.0,
            local_resolution_nm=0.1,
            local_size_nm=(2.0, 2.0),
        )

    def test_intercept_of_slower_target(self) -> None:
        intercept = predict_intercept((0.0, 0.0), (10.0, 0.0), (30.0, 0.0), 50.0)
        self.assertIsNotNone(intercept)
        assert intercept is not None
        self.assertAlmostEqual(intercept.time_hours, 0.5)
        self.assertEqual(intercept.point_nm, (25.0, 0.0))

    def test_safe_intervals_split_by_storm(self) -> None:
        first = self.weather_map.snapshot()
        self.weather_map.global_grid[0, 0] = THUNDERSTORM
        second = self.weather_map.snapshot()
        self.weather_map.global_grid[0, 0] = FREE
        third = self.weather_map.snapshot()
        intervals = safe_intervals((first, second, third), horizon_steps=2)
        self.assertEqual(intervals[(0, 0)], ((0, 0), (2, 2)))

    def test_planners_accept_immediate_goal(self) -> None:
        snapshot = self.weather_map.snapshot()
        frames = tuple(snapshot for _ in range(5))
        for planner in (
            SIPPPlanner(horizon_steps=4),
            TimeExpandedAStarPlanner(horizon_steps=4),
        ):
            result = planner.plan(frames, (1.5, 1.5), (1.5, 1.5), (0.0, 0.0))
            self.assertTrue(result.reached_goal)
            self.assertEqual(len(result.waypoints), 1)

    def test_sipp_prefers_earlier_intercept_over_waiting_at_start(self) -> None:
        snapshot = self.weather_map.snapshot()
        frames = tuple(snapshot for _ in range(11))
        result = SIPPPlanner(horizon_steps=10).plan(
            frames,
            (1.5, 1.5),
            (8.5, 1.5),
            (-60.0, 0.0),
        )
        self.assertTrue(result.reached_goal)
        self.assertGreater(len(result.waypoints), 1)
        self.assertLess(result.waypoints[-1].time_step, 7)

    def test_time_astar_uses_rendezvous_distance_tolerance(self) -> None:
        snapshot = self.weather_map.snapshot()
        frames = tuple(snapshot for _ in range(5))
        result = TimeExpandedAStarPlanner(
            horizon_steps=4,
            goal_tolerance_nm=1.0,
        ).plan(
            frames,
            (1.5, 1.5),
            (2.4, 1.5),
            (0.0, 0.0),
        )
        self.assertTrue(result.reached_goal)
        self.assertEqual(len(result.waypoints), 1)

    def test_time_astar_returns_marked_best_effort_path_at_expansion_limit(self) -> None:
        snapshot = self.weather_map.snapshot()
        frames = tuple(snapshot for _ in range(11))
        result = TimeExpandedAStarPlanner(
            horizon_steps=10,
            goal_tolerance_nm=0.1,
            maximum_expanded_states=5,
        ).plan(
            frames,
            (0.5, 0.5),
            (9.5, 9.5),
            (0.0, 0.0),
        )
        self.assertFalse(result.reached_goal)
        self.assertEqual(result.expanded_states, 5)
        self.assertIn("expansion limit reached", result.reason)
        self.assertGreater(len(result.waypoints), 1)
        self.assertTrue(math.isfinite(result.cost))
        start_distance = math.dist((0.5, 0.5), (9.5, 9.5))
        final = result.waypoints[-1]
        self.assertLess(math.dist((final.x_nm, final.y_nm), (9.5, 9.5)), start_distance)


if __name__ == "__main__":
    unittest.main()
