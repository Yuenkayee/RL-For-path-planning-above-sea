"""Training orchestration for the discrete maximum-entropy SAC baseline."""

from __future__ import annotations

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from algorithm.common import Transition
from algorithm.sac import SACAgent
from env.returnEnv import ReturnEnv

from .checkpoint import save_checkpoint
from .config import load_algorithm_config


def train_sac(
    env: ReturnEnv,
    *,
    episodes: int,
    max_steps_per_episode: int | None = None,
    config_path: str | Path = "config/sacConfig.json",
    checkpoint_path: str | Path | None = None,
    log_dir: str | Path | None = None,
    seed: int = 0,
) -> tuple[SACAgent, dict[str, float]]:
    observation, _ = env.reset(seed=seed)
    config = load_algorithm_config(config_path)
    agent = SACAgent(env.action_count, observation, seed=seed, **config)
    total_reward = 0.0
    total_steps = 0
    successes = 0
    writer = SummaryWriter(log_dir=str(log_dir)) if log_dir is not None else None
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        step = 0
        episode_reward = 0.0
        while True:
            action, _ = agent.predict(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.observe(Transition(observation, action, reward, next_observation, done))
            update_metrics = agent.update()
            if writer is not None and update_metrics["samples"]:
                writer.add_scalar("train/loss", update_metrics["loss"], total_steps)
                writer.add_scalar("train/alpha", update_metrics["alpha"], total_steps)
            total_reward += reward
            episode_reward += reward
            total_steps += 1
            step += 1
            observation = next_observation
            if done or (max_steps_per_episode is not None and step >= max_steps_per_episode):
                successes += info.get("outcome") == "success"
                if writer is not None:
                    writer.add_scalar("episode/reward", episode_reward, episode)
                break
    metrics = {
        "episodes": float(episodes),
        "steps": float(total_steps),
        "mean_reward": total_reward / max(1, episodes),
        "success_rate": successes / max(1, episodes),
    }
    if checkpoint_path is not None:
        save_checkpoint(agent, checkpoint_path, metadata=metrics)
    if writer is not None:
        writer.close()
    return agent, metrics


__all__ = ["train_sac"]
