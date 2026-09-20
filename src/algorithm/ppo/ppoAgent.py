"""PPO rollout evaluation, GAE calculation and clipped updates."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from algorithm.common.featureExtractor import (
    compress_observation,
    observations_to_tensors,
)
from algorithm.common.replayBuffer import RolloutBuffer, RolloutStep

from .policy import MaskedCategoricalPolicy
from .ppoNetwork import PPONetwork


class PPOAgent:
    def __init__(
        self,
        action_count: int,
        sample_observation: dict[str, Any],
        *,
        learning_rate: float = 3e-4,
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        value_coefficient: float = 0.5,
        entropy_coefficient: float = 0.01,
        batch_size: int = 64,
        update_epochs: int = 10,
        seed: int | None = None,
        device: str = "cpu",
    ) -> None:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.device = torch.device(device)
        history_frames = int(np.asarray(sample_observation["global_weather"]).shape[0])
        self.network = PPONetwork(history_frames, action_count).to(self.device)
        self.policy = MaskedCategoricalPolicy(self.network)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.buffer = RolloutBuffer()
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.batch_size = batch_size
        self.update_epochs = update_epochs

    def predict(
        self, observation: dict[str, Any], *, deterministic: bool = False
    ) -> tuple[int, dict[str, float]]:
        actions, predictions = self.predict_batch((observation,), deterministic=deterministic)
        return actions[0], predictions[0]

    def predict_batch(
        self,
        observations: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        *,
        deterministic: bool = False,
    ) -> tuple[list[int], list[dict[str, float]]]:
        """Select actions for several environments in one network forward pass."""
        if not observations:
            return [], []
        tensors = observations_to_tensors(observations, self.device)
        self.network.eval()
        with torch.no_grad():
            logits, values = self.network(tensors)
            distribution = Categorical(logits=logits)
            actions = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
            log_probabilities = distribution.log_prob(actions)
        return actions.cpu().tolist(), [
            {
                "log_probability": float(log_probability),
                "value": float(value),
            }
            for log_probability, value in zip(
                log_probabilities.cpu().tolist(), values.cpu().tolist(), strict=True
            )
        ]

    def predict_values(
        self, observations: list[dict[str, Any]] | tuple[dict[str, Any], ...]
    ) -> list[float]:
        """Return critic values used to bootstrap a fixed-length rollout."""
        if not observations:
            return []
        tensors = observations_to_tensors(observations, self.device)
        self.network.eval()
        with torch.no_grad():
            _, values = self.network(tensors)
        return [float(value) for value in values.cpu().tolist()]

    def observe(
        self,
        observation: dict[str, Any],
        action: int,
        reward: float,
        done: bool,
        prediction_info: dict[str, float],
        *,
        environment_id: int = 0,
    ) -> None:
        self.buffer.add(
            RolloutStep(
                compress_observation(observation),
                action,
                reward,
                prediction_info["value"],
                prediction_info["log_probability"],
                done,
                environment_id,
            )
        )

    def update(
        self,
        *,
        last_value: float = 0.0,
        last_values: dict[int, float] | None = None,
    ) -> dict[str, float]:
        if not self.buffer.steps:
            return {"loss": 0.0, "samples": 0.0}
        advantages = [0.0] * len(self.buffer.steps)
        returns = [0.0] * len(self.buffer.steps)
        indices_by_environment: dict[int, list[int]] = {}
        for index, step in enumerate(self.buffer.steps):
            indices_by_environment.setdefault(step.environment_id, []).append(index)
        bootstrap_values = last_values or {0: last_value}
        for environment_id, indices in indices_by_environment.items():
            gae = 0.0
            next_value = bootstrap_values.get(environment_id, 0.0)
            for index in reversed(indices):
                step = self.buffer.steps[index]
                continuation = 0.0 if step.done else 1.0
                delta = (
                    step.reward
                    + self.discount_factor * next_value * continuation
                    - step.value
                )
                gae = delta + self.discount_factor * self.gae_lambda * continuation * gae
                advantages[index] = gae
                returns[index] = gae + step.value
                next_value = step.value
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std(unbiased=False) + 1e-8
        )
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        old_log_probabilities = torch.as_tensor(
            [step.log_probability for step in self.buffer.steps],
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            [step.action for step in self.buffer.steps], dtype=torch.long, device=self.device
        )
        observations = [step.observation for step in self.buffer.steps]

        actor_total = value_total = entropy_total = 0.0
        update_count = 0
        self.network.train()
        for _ in range(self.update_epochs):
            permutation = torch.randperm(len(observations), device=self.device)
            for start in range(0, len(observations), self.batch_size):
                indices = permutation[start : start + self.batch_size]
                batch_observations = observations_to_tensors(
                    [observations[int(index)] for index in indices.cpu()], self.device
                )
                logits, values = self.network(batch_observations)
                distribution = Categorical(logits=logits)
                new_log_probabilities = distribution.log_prob(actions[indices])
                ratio = torch.exp(new_log_probabilities - old_log_probabilities[indices])
                unclipped = ratio * advantages_tensor[indices]
                clipped = (
                    torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                    * advantages_tensor[indices]
                )
                actor_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, returns_tensor[indices])
                entropy = distribution.entropy().mean()
                loss = (
                    actor_loss
                    + self.value_coefficient * value_loss
                    - self.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                actor_total += float(actor_loss.item())
                value_total += float(value_loss.item())
                entropy_total += float(entropy.item())
                update_count += 1

        sample_count = len(self.buffer.steps)
        self.buffer.clear()
        return {
            "loss": (actor_total + self.value_coefficient * value_total) / update_count,
            "actor_loss": actor_total / update_count,
            "value_loss": value_total / update_count,
            "entropy": entropy_total / update_count,
            "samples": float(sample_count),
        }

    def state_dict(self) -> dict:
        return {
            "algorithm": "ppo",
            "action_count": self.network.action_count,
            "action_semantics": "wait_plus_guidance_heading_residual_v1",
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        checkpoint_action_count = state.get("action_count")
        if checkpoint_action_count is None:
            actor_output = state.get("network", {}).get("actor.2.weight")
            if actor_output is not None:
                checkpoint_action_count = int(actor_output.shape[0])
        if checkpoint_action_count is not None and checkpoint_action_count != self.network.action_count:
            raise ValueError(
                "checkpoint action space is incompatible with the residual PPO action space: "
                f"checkpoint={checkpoint_action_count}, current={self.network.action_count}"
            )
        semantics = state.get("action_semantics")
        if semantics not in {None, "wait_plus_guidance_heading_residual_v1"}:
            raise ValueError(f"unsupported checkpoint action semantics: {semantics}")
        self.network.load_state_dict(state["network"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])


__all__ = ["PPOAgent"]
