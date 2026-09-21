from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from env.config import EnvironmentConfig
from env.returnEnv import ReturnEnv
from experiment import EXPERIMENT_GROUPS
from inference.episodeVisualization import run_episode_trace, save_episode_visualization
from model.weatherSystem import SimulationParameters, WeatherParameters
from training.curriculum import (
    DEFAULT_CURRICULUM,
    parameters_for_stage,
    stage_for_episode,
    stage_for_progress,
)
from training.evaluator import evaluate_policy, save_evaluation_plot


class _WaitPolicy:
    def predict(self, observation, *, deterministic=False):
        del observation, deterministic
        return 0, {}


def _easy_environment() -> ReturnEnv:
    parameters = SimulationParameters(
        weather=WeatherParameters(
            initial_storm_count=0,
            maximum_storm_count=0,
            storm_motion_speed_knots=0.0,
        ),
        total_time_minutes=2.0,
        helicopter_initial_nm=(5.01, 5.01),
        frigate_initial_nm=(5.01, 5.01),
    )
    return ReturnEnv(
        EnvironmentConfig(
            maximum_episode_minutes=2.0,
            frigate_heading_choices_deg=(0.0,),
            randomize_frigate_initial_position=False,
            weather_history_frames=2,
        ),
        weather_parameters=parameters,
    )


class ExperimentAndEvaluationTests(unittest.TestCase):
    def test_every_registered_config_exists(self) -> None:
        self.assertEqual(len(EXPERIMENT_GROUPS), 5)
        for group in EXPERIMENT_GROUPS:
            self.assertTrue((ROOT / group.config_file).is_file(), group.name)

    def test_curriculum_boundaries(self) -> None:
        self.assertEqual(stage_for_progress(0.0), DEFAULT_CURRICULUM[0])
        self.assertEqual(stage_for_progress(1.0), DEFAULT_CURRICULUM[-1])
        self.assertEqual(DEFAULT_CURRICULUM[-1].storm_area_scale, 1.5)
        expected = [
            DEFAULT_CURRICULUM[0],
            DEFAULT_CURRICULUM[1],
            DEFAULT_CURRICULUM[2],
            DEFAULT_CURRICULUM[3],
            DEFAULT_CURRICULUM[3],
        ]
        actual = [stage_for_episode(index, 1000) for index in (0, 250, 500, 750, 999)]
        self.assertEqual(actual, expected)

    def test_curriculum_stage_changes_weather_parameters(self) -> None:
        base = SimulationParameters.from_json()
        static = parameters_for_stage(base, DEFAULT_CURRICULUM[1]).weather
        dense = parameters_for_stage(base, DEFAULT_CURRICULUM[3]).weather

        self.assertEqual((static.initial_storm_count, static.maximum_storm_count), (2, 2))
        self.assertEqual(static.storm_motion_speed_knots, 0.0)
        self.assertEqual(static.storm_area_scale_range, (0.8, 0.8))
        self.assertEqual((dense.initial_storm_count, dense.maximum_storm_count), (8, 12))
        self.assertEqual(dense.storm_motion_speed_knots, base.weather.storm_motion_speed_knots)
        self.assertEqual(dense.storm_area_scale_range, (1.0, 1.5))

    def test_seed_matched_evaluator(self) -> None:
        result = evaluate_policy(_easy_environment, _WaitPolicy(), seeds=(1, 2))
        self.assertEqual(result.successes, 2)
        self.assertEqual(result.success_rate, 1.0)
        with tempfile.TemporaryDirectory() as directory:
            output = save_evaluation_plot({"wait": result}, Path(directory) / "result.png")
            self.assertGreater(output.stat().st_size, 0)

    def test_fixed_ppo_evaluation_scenarios_have_multiple_route_conflicts(self) -> None:
        ppo_config = json.loads((ROOT / "config/ppoConfig.json").read_text(encoding="utf-8"))
        parameters = parameters_for_stage(
            SimulationParameters.from_json(), DEFAULT_CURRICULUM[-1]
        )
        env = ReturnEnv(weather_parameters=parameters)
        minimum_conflicts = ppo_config["evaluation_minimum_route_conflicts"]
        for seed in ppo_config["evaluation_seeds"]:
            _, info = env.reset(
                seed=seed,
                options={"minimum_route_conflicts": minimum_conflicts},
            )
            self.assertGreaterEqual(
                info["episode_config"]["route_conflict_regions"],
                minimum_conflicts,
                f"seed={seed}",
            )

    def test_episode_trace_and_static_visualization(self) -> None:
        env = _easy_environment()
        trace = run_episode_trace(
            env,
            _WaitPolicy(),
            method="wait",
            seed=3,
            max_steps=2,
            frame_stride=1,
        )
        self.assertEqual(trace.outcome, "success")
        self.assertEqual(trace.steps, 1)
        self.assertEqual(len(trace.helicopter_path_nm), 2)
        with tempfile.TemporaryDirectory() as directory:
            output = save_episode_visualization(trace, Path(directory) / "episode.png")
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
