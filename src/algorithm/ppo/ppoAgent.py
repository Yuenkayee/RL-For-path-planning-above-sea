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
        tensors = observations_to_tensors((observation,), self.device)
        self.network.eval()
        with torch.no_grad():
            action, log_probability, value = self.policy.select(
                tensors, deterministic=deterministic
            )
        return action, {"log_probability": log_probability, "value": value}

    def observe(
        self,
        observation: dict[str, Any],
        action: int,
        reward: float,
        done: bool,
        prediction_info: dict[str, float],
    ) -> None:
        self.buffer.add(
            RolloutStep(
                compress_observation(observation),
                action,
                reward,
                prediction_info["value"],
                prediction_info["log_probability"],
                done,
            )
        )

    def update(self, *, last_value: float = 0.0) -> dict[str, float]:
        if not self.buffer.steps:
            return {"loss": 0.0, "samples": 0.0}
        advantages = [0.0] * len(self.buffer.steps)
        returns = [0.0] * len(self.buffer.steps)
        gae = 0.0
        next_value = last_value
        for index in reversed(range(len(self.buffer.steps))):
            step = self.buffer.steps[index]
            continuation = 0.0 if step.done else 1.0
            delta = step.reward + self.discount_factor * next_value * continuation - step.value
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
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.network.load_state_dict(state["network"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])


__all__ = ["PPOAgent"]
