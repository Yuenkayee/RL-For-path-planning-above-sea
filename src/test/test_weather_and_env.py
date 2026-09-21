from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.actionMask import build_action_mask, heading_for_residual_action
from env.config import EnvironmentConfig, PlannerConfig
from env.returnEnv import ReturnEnv
from env.reward import RewardWeights, calculate_reward, newly_reached_proximity_bonus
from env.termination import successful_rendezvous
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
    def test_default_training_scenario_configuration(self) -> None:
        env_config = EnvironmentConfig.from_json()
        weather_parameters = SimulationParameters.from_json()

        self.assertEqual(env_config.helicopter_speed_knots, (0.0, 100.0))
        self.assertEqual(
            env_config.frigate_heading_choices_deg,
            (315.0, 0.0, 45.0, 90.0, 135.0),
        )
        self.assertEqual(env_config.action_count, 6)
        self.assertEqual(env_config.success_distance_nm, 1.0)
        self.assertEqual(weather_parameters.map_size_nm, (80.0, 80.0))
        self.assertEqual(weather_parameters.frigate_initial_nm, (20.0, 20.0))
        self.assertEqual(weather_parameters.weather.initial_storm_count, 8)
        self.assertEqual(weather_parameters.weather.maximum_storm_count, 12)
        self.assertEqual(weather_parameters.weather.storm_area_scale_range, (1.0, 1.5))

    def test_residual_action_is_composed_with_planner_heading(self) -> None:
        offsets = (-45.0, -22.5, 0.0, 22.5, 45.0)
        self.assertIsNone(heading_for_residual_action(0, 225.0, offsets))
        self.assertEqual(heading_for_residual_action(3, 225.0, offsets), 225.0)
        self.assertEqual(heading_for_residual_action(5, 330.0, offsets), 15.0)

    def test_planner_defaults_use_long_horizon_and_frequent_replanning(self) -> None:
        config = PlannerConfig.from_json()
        self.assertEqual(config.horizon_steps, 90)
        self.assertEqual(config.replanning_interval_seconds, 12.0)
        self.assertEqual(config.lookahead_distance_nm, 3.0)

    def test_rendezvous_uses_one_nautical_mile_distance_not_grid_alignment(self) -> None:
        weather_map = WeatherMap(area_size_nm=(10.0, 10.0), local_origin_nm=(0.0, 0.0))
        self.assertTrue(
            successful_rendezvous(
                weather_map,
                (0.95, 0.5),
                (1.05, 0.5),
                maximum_distance_nm=1.0,
            )
        )
        self.assertFalse(
            successful_rendezvous(
                weather_map,
                (0.5, 0.5),
                (1.6, 0.5),
                maximum_distance_nm=1.0,
            )
        )

    def test_legacy_success_resolution_config_maps_to_distance(self) -> None:
        config = EnvironmentConfig.from_mapping({"success_grid_resolution_nm": 0.75})
        self.assertEqual(config.success_distance_nm, 0.75)
        self.assertEqual(config.success_grid_resolution_nm, 0.75)

    def test_proximity_rewards_are_awarded_once_per_threshold(self) -> None:
        weights = RewardWeights()
        reached: set[float] = set()

        bonus, newly_reached = newly_reached_proximity_bonus(weights, 4.0, reached)
        self.assertEqual((bonus, newly_reached), (5.0, (5.0,)))
        reached.update(newly_reached)

        bonus, newly_reached = newly_reached_proximity_bonus(weights, 4.5, reached)
        self.assertEqual((bonus, newly_reached), (0, ()))

        bonus, newly_reached = newly_reached_proximity_bonus(weights, 0.4, reached)
        self.assertEqual((bonus, newly_reached), (70.0, (2.0, 1.0, 0.5)))

    def test_planner_potential_shaping_is_added_to_reward(self) -> None:
        weights = RewardWeights(step=0.0, progress=0.0, planner_potential=2.0)
        reward = calculate_reward(
            weights,
            previous_distance_nm=10.0,
            current_distance_nm=10.0,
            waited=False,
            speed_switched=False,
            outcome=None,
            potential_shaping=0.75,
        )
        self.assertEqual(reward, 1.5)

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
            frigate_heading_choices_deg=(0.0,),
            randomize_frigate_initial_position=False,
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

    def test_weather_nowcast_does_not_mutate_live_system(self) -> None:
        parameters = replace(
            clear_weather_parameters(frigate_initial_nm=(20.0, 20.0)),
            weather=WeatherParameters(
                initial_storm_count=1,
                maximum_storm_count=1,
                storm_motion_speed_knots=16.0,
            ),
        )
        system = WeatherSystem(parameters)
        before = system.current_frame()
        forecasts = system.forecast_snapshots(3, step_minutes=1.0)
        after = system.current_frame()
        self.assertEqual(len(forecasts), 4)
        self.assertEqual(before, after)

    def test_episode_configuration_is_seeded_and_randomized(self) -> None:
        parameters = clear_weather_parameters(frigate_initial_nm=(20.0, 20.0))
        config = EnvironmentConfig(
            frigate_heading_choices_deg=(315.0, 0.0, 45.0, 90.0, 135.0),
        )
        env = ReturnEnv(config, weather_parameters=parameters)

        _, first = env.reset(seed=17)
        _, repeated = env.reset(seed=17)
        _, different = env.reset(seed=18)

        self.assertEqual(first["episode_config"], repeated["episode_config"])
        self.assertIn(
            first["episode_config"]["frigate_heading_deg"],
            config.frigate_heading_choices_deg,
        )
        self.assertGreaterEqual(first["episode_config"]["storm_area_scale"], 1.0)
        self.assertLessEqual(first["episode_config"]["storm_area_scale"], 1.5)
        self.assertNotEqual(
            first["episode_config"]["storm_area_scale"],
            different["episode_config"]["storm_area_scale"],
        )
        self.assertNotEqual(
            first["episode_config"]["frigate_initial_nm"],
            different["episode_config"]["frigate_initial_nm"],
        )
        frigate_start = different["episode_config"]["frigate_initial_nm"]
        heading_rad = math.radians(different["episode_config"]["frigate_heading_deg"])
        one_hour_end = (
            frigate_start[0] + config.frigate_speed_knots * math.sin(heading_rad),
            frigate_start[1] + config.frigate_speed_knots * math.cos(heading_rad),
        )
        margin = config.frigate_boundary_margin_nm
        self.assertTrue(
            margin <= one_hour_end[0] < parameters.map_size_nm[0] - margin
        )
        self.assertTrue(
            margin <= one_hour_end[1] < parameters.map_size_nm[1] - margin
        )
        self.assertGreaterEqual(
            math.dist(parameters.helicopter_initial_nm, frigate_start),
            config.frigate_minimum_initial_distance_nm,
        )
        self.assertEqual(
            env.weather_system.parameters.weather.storm_area_scale,
            different["episode_config"]["storm_area_scale"],
        )


if __name__ == "__main__":
    unittest.main()
