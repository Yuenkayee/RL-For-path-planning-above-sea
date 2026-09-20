from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithm.common import Transition
from algorithm.dqn import DQNAgent
from algorithm.ppo import PPOAgent
from algorithm.sac import SACAgent
from env.config import EnvironmentConfig
from env.returnEnv import ReturnEnv
from model.weatherSystem import SimulationParameters, WeatherParameters
from training.checkpoint import load_checkpoint, save_checkpoint
from training.trainPPO import _episode_seed, _flush_ordered_episode_metrics, train_ppo


class _RecordingWriter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float | int | bool, int]] = []

    def add_scalar(self, tag: str, value: float | int | bool, step: int) -> None:
        self.calls.append((tag, value, step))


def make_environment() -> ReturnEnv:
    parameters = SimulationParameters(
        weather=WeatherParameters(
            initial_storm_count=0,
            maximum_storm_count=0,
            storm_motion_speed_knots=0.0,
        ),
        total_time_minutes=5.0,
        helicopter_initial_nm=(5.0, 5.0),
        frigate_initial_nm=(20.0, 20.0),
        random_seed=3,
    )
    return ReturnEnv(
        EnvironmentConfig(maximum_episode_minutes=5.0, weather_history_frames=2),
        weather_parameters=parameters,
    )


def make_success_environment() -> ReturnEnv:
    parameters = SimulationParameters(
        weather=WeatherParameters(
            initial_storm_count=0,
            maximum_storm_count=0,
            storm_motion_speed_knots=0.0,
        ),
        total_time_minutes=2.0,
        helicopter_initial_nm=(5.01, 5.01),
        frigate_initial_nm=(5.01, 5.01),
        random_seed=3,
    )
    return ReturnEnv(
        EnvironmentConfig(
            maximum_episode_minutes=2.0,
            frigate_heading_choices_deg=(0.0,),
            success_distance_nm=10.0,
            weather_history_frames=2,
        ),
        weather_parameters=parameters,
    )


class AlgorithmTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = make_environment()
        self.observation, _ = self.env.reset(seed=3)
        self.next_observation, self.reward, terminated, truncated, _ = self.env.step(0)
        self.done = terminated or truncated

    def test_ppo_predict_update_and_checkpoint(self) -> None:
        agent = PPOAgent(self.env.action_count, self.observation, update_epochs=1, seed=1)
        action, info = agent.predict(self.observation)
        self.assertTrue(self.observation["action_mask"][action])
        agent.observe(self.observation, action, self.reward, self.done, info)
        metrics = agent.update()
        self.assertEqual(metrics["samples"], 1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(agent, Path(directory) / "ppo.pt")
            restored = PPOAgent(self.env.action_count, self.observation, seed=2)
            load_checkpoint(restored, path)
            for name, value in agent.network.state_dict().items():
                self.assertTrue(torch.equal(value, restored.network.state_dict()[name]))

    def test_parallel_ppo_reports_every_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "ppo.json"
            config_path.write_text(
                json.dumps(
                    {
                        "algorithm": "masked_ppo",
                        "learning_rate": 0.0003,
                        "discount_factor": 0.99,
                        "gae_lambda": 0.95,
                        "clip_range": 0.2,
                        "entropy_coefficient": 0.01,
                        "value_coefficient": 0.5,
                        "rollout_steps": 2,
                        "batch_size": 2,
                        "update_epochs": 1,
                    }
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                _, metrics = train_ppo(
                    make_success_environment(),
                    episodes=3,
                    max_steps_per_episode=5,
                    config_path=config_path,
                    seed=10,
                    num_envs=2,
                    device="cpu",
                    progress_interval_steps=1,
                )

        self.assertEqual(metrics["episodes"], 3.0)
        self.assertEqual(metrics["steps"], 3.0)
        self.assertEqual(metrics["num_envs"], 2.0)
        self.assertEqual(metrics["success_rate"], 1.0)
        self.assertEqual(metrics["curriculum"], "enabled")
        self.assertIn("curriculum=interception_only", output.getvalue())
        completed_lines = [
            line
            for line in output.getvalue().splitlines()
            if line.startswith("[PPO] episode") and " complete " in line
        ]
        self.assertEqual(len(completed_lines), 3)

    def test_hard_seed_replay_is_deterministic(self) -> None:
        first = [_episode_seed(index, 10, (1, 4, 6), 1.0) for index in range(6)]
        second = [_episode_seed(index, 10, (1, 4, 6), 1.0) for index in range(6)]
        self.assertEqual(first, second)
        self.assertTrue(all(replayed for _, replayed in first))
        self.assertTrue(all(seed in {1, 4, 6} for seed, _ in first))

    def test_parallel_episode_metrics_are_flushed_in_episode_order(self) -> None:
        writer = _RecordingWriter()
        pending = {
            2: (12.0, 30, True, 1),
            0: (-5.0, 20, False, 0),
        }
        next_episode = _flush_ordered_episode_metrics(writer, pending, 0)  # type: ignore[arg-type]
        self.assertEqual(next_episode, 1)
        self.assertEqual({call[2] for call in writer.calls}, {0})

        pending[1] = (3.0, 25, True, 0)
        next_episode = _flush_ordered_episode_metrics(  # type: ignore[arg-type]
            writer, pending, next_episode
        )
        self.assertEqual(next_episode, 3)
        self.assertEqual(
            [call[2] for call in writer.calls],
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
        )
        self.assertFalse(pending)

    def test_dqn_updates_from_replay(self) -> None:
        agent = DQNAgent(
            self.env.action_count,
            self.observation,
            batch_size=1,
            n_step=1,
            target_update_interval=1,
            seed=1,
        )
        agent.observe(
            Transition(self.observation, 0, self.reward, self.next_observation, self.done)
        )
        metrics = agent.update()
        self.assertEqual(metrics["samples"], 1.0)
        self.assertGreaterEqual(metrics["loss"], 0.0)

    def test_discrete_sac_updates_from_replay(self) -> None:
        agent = SACAgent(self.env.action_count, self.observation, batch_size=1, seed=1)
        agent.observe(
            Transition(self.observation, 0, self.reward, self.next_observation, self.done)
        )
        metrics = agent.update()
        self.assertEqual(metrics["samples"], 1.0)
        self.assertGreater(metrics["alpha"], 0.0)


if __name__ == "__main__":
    unittest.main()
