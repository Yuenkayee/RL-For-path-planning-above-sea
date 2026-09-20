"""Parallel training orchestration for the global-guided Masked PPO method."""

from __future__ import annotations

import functools
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv
from torch.utils.tensorboard import SummaryWriter

from algorithm.ppo import PPOAgent
from env.config import EnvironmentConfig, PlannerConfig
from env.returnEnv import ReturnEnv
from model.weatherSystem import SimulationParameters

from .checkpoint import save_checkpoint
from .config import load_algorithm_config
from .curriculum import DEFAULT_CURRICULUM, parameters_for_stage, stage_for_episode


class _EpisodeSequenceEnv(gym.Wrapper):
    """Assign deterministic episode ids, seeds and curriculum stages to a worker."""

    def __init__(
        self,
        env: ReturnEnv,
        episode_ids: Sequence[int],
        *,
        total_episodes: int,
        base_seed: int,
        base_weather_parameters: SimulationParameters,
        use_curriculum: bool,
        hard_episode_seeds: tuple[int, ...],
        hard_seed_replay_probability: float,
    ) -> None:
        if not episode_ids:
            raise ValueError("each worker must receive at least one episode id")
        super().__init__(env)
        self._episode_ids = tuple(int(episode_id) for episode_id in episode_ids)
        self._total_episodes = total_episodes
        self._base_seed = base_seed
        self._base_weather_parameters = base_weather_parameters
        self._use_curriculum = use_curriculum
        self._hard_episode_seeds = hard_episode_seeds
        self._hard_seed_replay_probability = hard_seed_replay_probability
        self._seed_cursor = 0
        self._active = False
        self._last_observation: dict[str, Any] | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del seed
        if self._seed_cursor >= len(self._episode_ids):
            self._active = False
            if self._last_observation is None:
                raise RuntimeError("worker became inactive before its first reset")
            return self._last_observation, {"inactive": True}
        episode_id = self._episode_ids[self._seed_cursor]
        self._seed_cursor += 1
        stage = stage_for_episode(episode_id, self._total_episodes)
        self.env._base_weather_parameters = (
            parameters_for_stage(self._base_weather_parameters, stage)
            if self._use_curriculum
            else self._base_weather_parameters
        )
        episode_seed, hard_seed_replay = _episode_seed(
            episode_id,
            self._base_seed,
            self._hard_episode_seeds,
            self._hard_seed_replay_probability,
        )
        observation, info = self.env.reset(seed=episode_seed, options=options)
        info["episode_id"] = episode_id
        info["curriculum_stage"] = stage.name if self._use_curriculum else "disabled"
        info["hard_seed_replay"] = hard_seed_replay
        self._last_observation = observation
        self._active = True
        return observation, info

    def step(self, action: int):
        if not self._active:
            if self._last_observation is None:
                raise RuntimeError("inactive worker has no observation")
            return self._last_observation, 0.0, False, False, {"inactive": True}
        return self.env.step(action)


def _make_episode_environment(
    env_config: EnvironmentConfig,
    weather_parameters: SimulationParameters,
    planner_config: PlannerConfig,
    episode_ids: tuple[int, ...],
    total_episodes: int,
    base_seed: int,
    use_curriculum: bool,
    hard_episode_seeds: tuple[int, ...],
    hard_seed_replay_probability: float,
) -> _EpisodeSequenceEnv:
    return _EpisodeSequenceEnv(
        ReturnEnv(
            env_config,
            planner_config=planner_config,
            weather_parameters=weather_parameters,
        ),
        episode_ids,
        total_episodes=total_episodes,
        base_seed=base_seed,
        base_weather_parameters=weather_parameters,
        use_curriculum=use_curriculum,
        hard_episode_seeds=hard_episode_seeds,
        hard_seed_replay_probability=hard_seed_replay_probability,
    )


def _episode_seed(
    episode_id: int,
    base_seed: int,
    hard_episode_seeds: tuple[int, ...],
    replay_probability: float,
) -> tuple[int, bool]:
    """Return a deterministic mixture of fresh and known-difficult scenarios."""
    rng = random.Random((base_seed + 1) * 1_000_003 + episode_id)
    replay = bool(hard_episode_seeds) and rng.random() < replay_probability
    if replay:
        return rng.choice(hard_episode_seeds), True
    return base_seed + episode_id, False


def _split_observations(
    batched_observation: dict[str, np.ndarray], number_of_environments: int
) -> list[dict[str, Any]]:
    return [
        {name: values[index] for name, values in batched_observation.items()}
        for index in range(number_of_environments)
    ]


def _final_info_at(infos: dict[str, Any], index: int) -> dict[str, Any]:
    values = infos.get("final_info")
    mask = infos.get("_final_info")
    if values is None or (mask is not None and not bool(mask[index])):
        return {}
    if isinstance(values, dict):
        result: dict[str, Any] = {}
        for name, batched_value in values.items():
            if name.startswith("_"):
                continue
            value_mask = values.get(f"_{name}")
            if value_mask is not None and not bool(value_mask[index]):
                continue
            if isinstance(batched_value, dict):
                result[name] = _final_info_at(
                    {"final_info": batched_value, "_final_info": value_mask}, index
                )
            else:
                result[name] = batched_value[index]
        return result
    value = values[index]
    return value if isinstance(value, dict) else {}


def _flush_ordered_episode_metrics(
    writer: SummaryWriter,
    pending: dict[int, tuple[float, int, bool, int]],
    next_episode_id: int,
) -> int:
    """Write all contiguous completed episodes in episode-id order."""
    while next_episode_id in pending:
        reward, steps, succeeded, curriculum_stage = pending.pop(next_episode_id)
        writer.add_scalar("episode/reward", reward, next_episode_id)
        writer.add_scalar("episode/steps", steps, next_episode_id)
        writer.add_scalar("episode/success", succeeded, next_episode_id)
        writer.add_scalar("curriculum/stage", curriculum_stage, next_episode_id)
        next_episode_id += 1
    return next_episode_id


def _resolve_device(requested: str) -> str:
    normalized = requested.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    if normalized == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested, but PyTorch cannot use Apple Metal")
    if normalized not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    return normalized


def _bootstrap_values(
    agent: PPOAgent,
    observations: list[dict[str, Any]],
    active_workers: list[int],
) -> dict[int, float]:
    values = agent.predict_values([observations[index] for index in active_workers])
    return dict(zip(active_workers, values, strict=True))


def train_ppo(
    env: ReturnEnv,
    *,
    episodes: int,
    max_steps_per_episode: int | None = None,
    config_path: str | Path = "config/ppoConfig.json",
    checkpoint_path: str | Path | None = None,
    log_dir: str | Path | None = None,
    seed: int = 0,
    num_envs: int = 1,
    device: str = "cpu",
    progress_interval_steps: int = 100,
    show_progress: bool = True,
    use_curriculum: bool = True,
) -> tuple[PPOAgent, dict[str, float | str]]:
    """Train PPO with batched policy inference and parallel environments."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if max_steps_per_episode is not None and max_steps_per_episode <= 0:
        raise ValueError("max_steps_per_episode must be positive")
    if progress_interval_steps <= 0:
        raise ValueError("progress_interval_steps must be positive")

    config = load_algorithm_config(config_path)
    hard_episode_seeds = tuple(int(value) for value in config.pop("hard_episode_seeds", ()))
    hard_seed_replay_probability = float(
        config.pop("hard_seed_replay_probability", 0.0)
    )
    if not 0.0 <= hard_seed_replay_probability <= 1.0:
        raise ValueError("hard_seed_replay_probability must lie in [0, 1]")
    rollout_steps = int(config["rollout_steps"])
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    selected_device = _resolve_device(device)
    worker_count = min(num_envs, episodes)

    sample_observation, _ = env.reset(seed=seed)
    agent = PPOAgent(
        env.action_count,
        sample_observation,
        learning_rate=config["learning_rate"],
        discount_factor=config["discount_factor"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        value_coefficient=config["value_coefficient"],
        entropy_coefficient=config["entropy_coefficient"],
        batch_size=config["batch_size"],
        update_epochs=config["update_epochs"],
        seed=seed,
        device=selected_device,
        residual_heading_offsets_deg=env.config.residual_heading_offsets_deg,
    )

    episode_ids_by_worker = [
        tuple(range(index, episodes, worker_count)) for index in range(worker_count)
    ]
    environment_factories = [
        functools.partial(
            _make_episode_environment,
            env.config,
            env._base_weather_parameters,
            env.planner_config,
            tuple(episode_ids),
            episodes,
            seed,
            use_curriculum,
            hard_episode_seeds,
            hard_seed_replay_probability,
        )
        for episode_ids in episode_ids_by_worker
    ]
    vector_env_type = AsyncVectorEnv if worker_count > 1 else SyncVectorEnv
    vector_env = vector_env_type(
        environment_factories,
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    writer = SummaryWriter(log_dir=str(log_dir)) if log_dir is not None else None

    completed_episodes = 0
    total_reward = 0.0
    total_steps = 0
    successes = 0
    update_count = 0
    episode_positions = [0] * worker_count
    episode_steps = [0] * worker_count
    episode_rewards = [0.0] * worker_count
    episode_started_at = [time.perf_counter()] * worker_count
    pending_episode_metrics: dict[int, tuple[float, int, bool, int]] = {}
    next_episode_to_log = 0
    maximum_progress_steps = max_steps_per_episode or round(
        env.config.maximum_episode_minutes * 60.0 / env.config.control_step_seconds
    )

    if show_progress:
        print(
            f"[PPO] starting {episodes} episodes with {worker_count} environment(s) "
            f"on {selected_device}; rollout_steps={rollout_steps}",
            flush=True,
        )
        for worker, episode_ids in enumerate(episode_ids_by_worker):
            episode_id = episode_ids[0]
            episode_seed, hard_replay = _episode_seed(
                episode_id,
                seed,
                hard_episode_seeds,
                hard_seed_replay_probability,
            )
            curriculum_name = (
                stage_for_episode(episode_id, episodes).name if use_curriculum else "disabled"
            )
            print(
                f"[PPO] episode {episode_id + 1}/{episodes} started "
                f"(worker={worker}, seed={episode_seed}, curriculum={curriculum_name}, "
                f"hard_replay={hard_replay})",
                flush=True,
            )

    try:
        batched_observation, _ = vector_env.reset()
        observations = _split_observations(batched_observation, worker_count)

        while completed_episodes < episodes:
            active_workers = [
                index
                for index in range(worker_count)
                if episode_positions[index] < len(episode_ids_by_worker[index])
            ]
            selected_actions, predictions = agent.predict_batch(
                [observations[index] for index in active_workers]
            )
            actions = np.zeros(worker_count, dtype=np.int64)
            prediction_by_worker: dict[int, dict[str, float]] = {}
            for worker, action, prediction in zip(
                active_workers, selected_actions, predictions, strict=True
            ):
                actions[worker] = action
                prediction_by_worker[worker] = prediction

            next_batch, rewards, terminated, truncated, infos = vector_env.step(actions)
            next_observations = _split_observations(next_batch, worker_count)
            forced_reset = np.zeros(worker_count, dtype=np.bool_)

            for worker in active_workers:
                episode_steps[worker] += 1
                reward = float(rewards[worker])
                episode_rewards[worker] += reward
                total_reward += reward
                total_steps += 1
                environment_done = bool(terminated[worker] or truncated[worker])
                reached_step_limit = bool(
                    max_steps_per_episode is not None
                    and episode_steps[worker] >= max_steps_per_episode
                )
                done = environment_done or reached_step_limit
                forced_reset[worker] = reached_step_limit and not environment_done
                agent.observe(
                    observations[worker],
                    int(actions[worker]),
                    reward,
                    done,
                    prediction_by_worker[worker],
                    environment_id=worker,
                )

                if not done:
                    if show_progress and episode_steps[worker] % progress_interval_steps == 0:
                        episode_id = episode_ids_by_worker[worker][episode_positions[worker]]
                        percent = min(
                            100.0, 100.0 * episode_steps[worker] / maximum_progress_steps
                        )
                        print(
                            f"[PPO] episode {episode_id + 1}/{episodes} progress "
                            f"{episode_steps[worker]}/{maximum_progress_steps} ({percent:.1f}%) "
                            f"reward={episode_rewards[worker]:.3f} worker={worker}",
                            flush=True,
                        )
                    continue

                episode_id = episode_ids_by_worker[worker][episode_positions[worker]]
                final_info = _final_info_at(infos, worker) if environment_done else {}
                outcome = final_info.get("outcome") or (
                    "max_steps" if reached_step_limit else "terminated"
                )
                successes += outcome == "success"
                completed_episodes += 1
                elapsed = time.perf_counter() - episode_started_at[worker]
                if writer is not None:
                    curriculum_stage = (
                        DEFAULT_CURRICULUM.index(stage_for_episode(episode_id, episodes))
                        if use_curriculum
                        else -1
                    )
                    pending_episode_metrics[episode_id] = (
                        episode_rewards[worker],
                        episode_steps[worker],
                        outcome == "success",
                        curriculum_stage,
                    )
                    next_episode_to_log = _flush_ordered_episode_metrics(
                        writer,
                        pending_episode_metrics,
                        next_episode_to_log,
                    )
                if show_progress:
                    print(
                        f"[PPO] episode {episode_id + 1}/{episodes} complete "
                        f"({100.0 * completed_episodes / episodes:.1f}% total) "
                        f"steps={episode_steps[worker]} reward={episode_rewards[worker]:.3f} "
                        f"outcome={outcome} elapsed={elapsed:.1f}s worker={worker}",
                        flush=True,
                    )

                episode_positions[worker] += 1
                episode_steps[worker] = 0
                episode_rewards[worker] = 0.0
                episode_started_at[worker] = time.perf_counter()
                if (
                    show_progress
                    and episode_positions[worker] < len(episode_ids_by_worker[worker])
                ):
                    next_episode_id = episode_ids_by_worker[worker][episode_positions[worker]]
                    next_seed, hard_replay = _episode_seed(
                        next_episode_id,
                        seed,
                        hard_episode_seeds,
                        hard_seed_replay_probability,
                    )
                    curriculum_name = (
                        stage_for_episode(next_episode_id, episodes).name
                        if use_curriculum
                        else "disabled"
                    )
                    print(
                        f"[PPO] episode {next_episode_id + 1}/{episodes} started "
                        f"(worker={worker}, seed={next_seed}, "
                        f"curriculum={curriculum_name}, hard_replay={hard_replay})",
                        flush=True,
                    )

            if np.any(forced_reset):
                next_batch, _ = vector_env.reset(options={"reset_mask": forced_reset})
                next_observations = _split_observations(next_batch, worker_count)
            observations = next_observations

            if len(agent.buffer) >= rollout_steps:
                still_active = [
                    index
                    for index in range(worker_count)
                    if episode_positions[index] < len(episode_ids_by_worker[index])
                ]
                update_started_at = time.perf_counter()
                if show_progress:
                    print(
                        f"[PPO] update {update_count + 1} started "
                        f"with {len(agent.buffer)} rollout samples",
                        flush=True,
                    )
                update_metrics = agent.update(
                    last_values=_bootstrap_values(agent, observations, still_active)
                )
                update_count += 1
                if show_progress:
                    print(
                        f"[PPO] update {update_count} complete "
                        f"loss={update_metrics['loss']:.5f} "
                        f"elapsed={time.perf_counter() - update_started_at:.1f}s",
                        flush=True,
                    )
                if writer is not None:
                    writer.add_scalar("train/loss", update_metrics["loss"], update_count)
                    writer.add_scalar("train/samples", update_metrics["samples"], update_count)

        if len(agent.buffer):
            update_started_at = time.perf_counter()
            if show_progress:
                print(
                    f"[PPO] final update {update_count + 1} started "
                    f"with {len(agent.buffer)} rollout samples",
                    flush=True,
                )
            update_metrics = agent.update(last_values={})
            update_count += 1
            if show_progress:
                print(
                    f"[PPO] final update {update_count} complete "
                    f"loss={update_metrics['loss']:.5f} "
                    f"elapsed={time.perf_counter() - update_started_at:.1f}s",
                    flush=True,
                )
            if writer is not None:
                writer.add_scalar("train/loss", update_metrics["loss"], update_count)
                writer.add_scalar("train/samples", update_metrics["samples"], update_count)
    finally:
        vector_env.close()
        if writer is not None:
            writer.close()

    metrics: dict[str, float | str] = {
        "episodes": float(episodes),
        "steps": float(total_steps),
        "mean_reward": total_reward / episodes,
        "success_rate": successes / episodes,
        "updates": float(update_count),
        "num_envs": float(worker_count),
        "device": selected_device,
        "curriculum": "enabled" if use_curriculum else "disabled",
        "hard_seed_replay_probability": hard_seed_replay_probability,
    }
    if checkpoint_path is not None:
        save_checkpoint(agent, checkpoint_path, metadata=metrics)
    return agent, metrics


__all__ = ["train_ppo"]
