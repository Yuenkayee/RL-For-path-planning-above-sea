from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from env.config import EnvironmentConfig
from env.returnEnv import ReturnEnv
from experiment import EXPERIMENT_GROUPS
from model.weatherSystem import SimulationParameters, WeatherParameters
from training.curriculum import DEFAULT_CURRICULUM, stage_for_progress
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
            frigate_heading_deg=0.0,
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

    def test_seed_matched_evaluator(self) -> None:
        result = evaluate_policy(_easy_environment, _WaitPolicy(), seeds=(1, 2))
        self.assertEqual(result.successes, 2)
        self.assertEqual(result.success_rate, 1.0)
        with tempfile.TemporaryDirectory() as directory:
            output = save_evaluation_plot({"wait": result}, Path(directory) / "result.png")
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
