from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.actionMask import build_action_mask
from env.config import EnvironmentConfig
from env.returnEnv import ReturnEnv
from model.weatherMap import THUNDERSTORM, WeatherMap
from model.weatherSystem import SimulationParameters, WeatherParameters, WeatherSystem


def clear_weather_parameters(**changes):
    weather = WeatherParameters(
        initial_storm_count=0,
        maximum_storm_count=0,
        storm_motion_speed_knots=0.0,
    )
    parameters = SimulationParameters(
        weather=weather,
        total_time_minutes=5.0,
        helicopter_initial_nm=(5.01, 5.01),
        frigate_initial_nm=(5.01, 5.01),
        random_seed=1,
    )
    return replace(parameters, **changes)


class WeatherAndEnvironmentTests(unittest.TestCase):
    def test_local_grid_reference_survives_window_move(self) -> None:
        system = WeatherSystem(clear_weather_parameters())
        local_grid = system.weather_map.local_grid
        system.move_local_window((20.0, 20.0))
        self.assertIs(local_grid, system.weather_map.local_grid)
        self.assertEqual(system.weather_map.local_origin_nm, (15.0, 15.0))

    def test_wait_is_masked_inside_storm(self) -> None:
        weather_map = WeatherMap(local_origin_nm=(0.0, 0.0))
        weather_map.set_weather_at(5.0, 5.0, THUNDERSTORM)
        mask = build_action_mask(
            weather_map,
            (5.0, 5.0),
            heading_count=16,
            flight_speed_knots=50.0,
            wait_speed_knots=0.0,
            duration_seconds=6.0,
        )
        self.assertFalse(any(mask))

    def test_wait_can_complete_rendezvous(self) -> None:
        config = EnvironmentConfig(
            maximum_episode_minutes=5.0,
            frigate_heading_deg=0.0,
            weather_history_frames=2,
        )
        env = ReturnEnv(config, weather_parameters=clear_weather_parameters())
        observation, _ = env.reset(seed=1)
        self.assertTrue(env.observation_space.contains(observation))
        _, _, terminated, truncated, info = env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["outcome"], "success")

    def test_weather_updates_after_ten_control_steps(self) -> None:
        parameters = clear_weather_parameters(
            frigate_initial_nm=(20.0, 20.0),
        )
        env = ReturnEnv(
            EnvironmentConfig(maximum_episode_minutes=5.0), weather_parameters=parameters
        )
        env.reset(seed=1)
        updates = []
        for _ in range(10):
            _, _, terminated, truncated, info = env.step(0)
            self.assertFalse(terminated or truncated)
            updates.append(info["weather_updated"])
        self.assertEqual(updates, [False] * 9 + [True])


if __name__ == "__main__":
    unittest.main()
