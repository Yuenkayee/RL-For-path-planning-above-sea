from __future__ import annotations

import sys
import tempfile
import unittest
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
