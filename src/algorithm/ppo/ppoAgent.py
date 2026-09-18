"""PPO rollout evaluation, GAE calculation and clipped updates."""

from __future__ import annotations

import math
from typing import Any

from algorithm.common.featureExtractor import DualResolutionFeatureExtractor
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
        update_epochs: int = 10,
        seed: int | None = None,
    ) -> None:
        self.extractor = DualResolutionFeatureExtractor()
        features = self.extractor(sample_observation)
        self.network = PPONetwork(len(features), action_count, seed=seed)
        self.policy = MaskedCategoricalPolicy(self.network, seed=seed)
        self.buffer = RolloutBuffer()
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_coefficient = value_coefficient
        self.update_epochs = update_epochs

    def predict(
        self, observation: dict[str, Any], *, deterministic: bool = False
    ) -> tuple[int, dict[str, float]]:
        features = self.extractor(observation)
        action, log_probability, value = self.policy.select(
            features, observation["action_mask"], deterministic=deterministic
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
                self.extractor.compact(observation),
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
        mean = sum(advantages) / len(advantages)
        variance = sum((value - mean) ** 2 for value in advantages) / len(advantages)
        scale = math.sqrt(variance + 1e-8)
        advantages = [(value - mean) / scale for value in advantages]

        actor_loss = 0.0
        value_loss = 0.0
        for _ in range(self.update_epochs):
            for step, advantage, target_return in zip(self.buffer.steps, advantages, returns):
                features = self.extractor(step.observation)
                probabilities = self.network.probabilities(features, step.observation["action_mask"])
                new_log_probability = math.log(max(probabilities[step.action], 1e-12))
                ratio = math.exp(new_log_probability - step.log_probability)
                clipped = min(max(ratio, 1.0 - self.clip_range), 1.0 + self.clip_range)
                actor_loss -= min(ratio * advantage, clipped * advantage)
                clipped_region = (advantage >= 0 and ratio > 1.0 + self.clip_range) or (
                    advantage < 0 and ratio < 1.0 - self.clip_range
                )
                if not clipped_region:
                    coefficient = self.learning_rate * ratio * advantage
                    for action_index, probability in enumerate(probabilities):
                        direction = (1.0 if action_index == step.action else 0.0) - probability
                        weights = self.network.actor_weights[action_index]
                        for feature_index, feature in enumerate(features):
                            weights[feature_index] += coefficient * direction * feature

                value = self.network.value(features)
                error = target_return - value
                value_loss += error * error
                for feature_index, feature in enumerate(features):
                    self.network.value_weights[feature_index] += (
                        self.learning_rate * self.value_coefficient * error * feature
                    )
        sample_count = len(self.buffer.steps) * self.update_epochs
        self.buffer.clear()
        return {
            "loss": (actor_loss + self.value_coefficient * value_loss) / sample_count,
            "actor_loss": actor_loss / sample_count,
            "value_loss": value_loss / sample_count,
            "samples": float(sample_count),
        }

    def state_dict(self) -> dict:
        return {"algorithm": "ppo", "network": self.network.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.network.load_state_dict(state["network"])


__all__ = ["PPOAgent"]
