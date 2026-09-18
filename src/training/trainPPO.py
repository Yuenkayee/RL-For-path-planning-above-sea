"""Training orchestration for the global-guided Masked PPO main method."""

from __future__ import annotations

from pathlib import Path

from algorithm.ppo import PPOAgent
from env.returnEnv import ReturnEnv

from .checkpoint import save_checkpoint
from .config import load_algorithm_config


def train_ppo(
    env: ReturnEnv,
    *,
    episodes: int,
    max_steps_per_episode: int | None = None,
    config_path: str | Path = "config/ppoConfig.json",
    checkpoint_path: str | Path | None = None,
    seed: int = 0,
) -> tuple[PPOAgent, dict[str, float]]:
    observation, _ = env.reset(seed=seed)
    config = load_algorithm_config(config_path)
    agent = PPOAgent(
        env.action_count,
        observation,
        learning_rate=config["learning_rate"],
        discount_factor=config["discount_factor"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        value_coefficient=config["value_coefficient"],
        update_epochs=config["update_epochs"],
        seed=seed,
    )
    total_reward = 0.0
    total_steps = 0
    successes = 0
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        step = 0
        while True:
            action, prediction = agent.predict(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.observe(observation, action, reward, done, prediction)
            total_reward += reward
            total_steps += 1
            step += 1
            observation = next_observation
            if done or (max_steps_per_episode is not None and step >= max_steps_per_episode):
                successes += info.get("outcome") == "success"
                break
        agent.update(last_value=0.0)
    metrics = {
        "episodes": float(episodes),
        "steps": float(total_steps),
        "mean_reward": total_reward / max(1, episodes),
        "success_rate": successes / max(1, episodes),
    }
    if checkpoint_path is not None:
        save_checkpoint(agent, checkpoint_path, metadata=metrics)
    return agent, metrics


__all__ = ["train_ppo"]
