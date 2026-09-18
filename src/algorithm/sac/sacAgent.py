"""Discrete maximum-entropy Soft Actor-Critic comparison agent."""

from __future__ import annotations

import math
import random
from typing import Any

from algorithm.common.featureExtractor import DualResolutionFeatureExtractor
from algorithm.common.mathUtils import argmax_masked, sample_categorical
from algorithm.common.replayBuffer import ReplayBuffer, Transition

from .sacNetwork import SACNetwork


class SACAgent:
    def __init__(
        self,
        action_count: int,
        sample_observation: dict[str, Any],
        *,
        learning_rate: float = 3e-4,
        discount_factor: float = 0.99,
        soft_update_factor: float = 0.005,
        replay_capacity: int = 1000000,
        batch_size: int = 256,
        automatic_entropy_tuning: bool = True,
        entropy_temperature: float = 0.2,
        seed: int | None = None,
    ) -> None:
        self.extractor = DualResolutionFeatureExtractor()
        feature_count = len(self.extractor(sample_observation))
        self.network = SACNetwork(feature_count, action_count, seed=seed)
        self.target = SACNetwork(feature_count, action_count, seed=seed)
        self.target.load_state_dict(self.network.state_dict())
        self.replay = ReplayBuffer(replay_capacity, seed=seed)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.soft_update_factor = soft_update_factor
        self.batch_size = batch_size
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.alpha = entropy_temperature
        self._rng = random.Random(seed)

    def predict(
        self, observation: dict[str, Any], *, deterministic: bool = False
    ) -> tuple[int, dict[str, float]]:
        features = self.extractor(observation)
        probabilities = self.network.probabilities(features, observation["action_mask"])
        action = (
            argmax_masked(probabilities, observation["action_mask"])
            if deterministic
            else sample_categorical(probabilities, self._rng)
        )
        return action, {"probability": probabilities[action]}

    def observe(self, transition: Transition) -> None:
        self.replay.add(
            Transition(
                self.extractor.compact(transition.observation),
                transition.action,
                transition.reward,
                self.extractor.compact(transition.next_observation),
                transition.done,
            )
        )

    def update(self) -> dict[str, float]:
        if len(self.replay) < self.batch_size:
            return {"loss": 0.0, "samples": 0.0, "alpha": self.alpha}
        critic_losses = []
        entropies = []
        for _, transition in self.replay.sample(self.batch_size):
            features = self.extractor(transition.observation)
            if transition.done:
                target_q = transition.reward
            else:
                next_features = self.extractor(transition.next_observation)
                next_probabilities = self.network.probabilities(
                    next_features, transition.next_observation["action_mask"]
                )
                target_q1 = self.target.q_values(next_features, 1)
                target_q2 = self.target.q_values(next_features, 2)
                soft_value = sum(
                    probability
                    * (
                        min(target_q1[action], target_q2[action])
                        - self.alpha * math.log(max(probability, 1e-12))
                    )
                    for action, probability in enumerate(next_probabilities)
                    if probability > 0
                )
                target_q = transition.reward + self.discount_factor * soft_value

            errors = []
            for critic, weights in ((1, self.network.q1_weights), (2, self.network.q2_weights)):
                prediction = self.network.q_values(features, critic)[transition.action]
                error = target_q - prediction
                errors.append(error)
                for index, feature in enumerate(features):
                    weights[transition.action][index] += self.learning_rate * error * feature
            critic_losses.append(sum(error * error for error in errors) / 2.0)

            probabilities = self.network.probabilities(features, transition.observation["action_mask"])
            q1 = self.network.q_values(features, 1)
            q2 = self.network.q_values(features, 2)
            terms = [
                self.alpha * (math.log(max(probability, 1e-12)) + 1.0)
                - min(q1[action], q2[action])
                for action, probability in enumerate(probabilities)
            ]
            expected_term = sum(p * term for p, term in zip(probabilities, terms))
            for action, probability in enumerate(probabilities):
                if probability == 0:
                    continue
                gradient = probability * (terms[action] - expected_term)
                for index, feature in enumerate(features):
                    self.network.actor_weights[action][index] -= self.learning_rate * gradient * feature
            entropy = -sum(
                probability * math.log(max(probability, 1e-12))
                for probability in probabilities
                if probability > 0
            )
            entropies.append(entropy)

        tau = self.soft_update_factor
        for target_rows, source_rows in (
            (self.target.q1_weights, self.network.q1_weights),
            (self.target.q2_weights, self.network.q2_weights),
        ):
            for target_row, source_row in zip(target_rows, source_rows):
                for index, source in enumerate(source_row):
                    target_row[index] = (1.0 - tau) * target_row[index] + tau * source
        if self.automatic_entropy_tuning:
            target_entropy = 0.8 * math.log(self.network.action_count)
            mean_entropy = sum(entropies) / len(entropies)
            self.alpha = max(
                1e-4,
                self.alpha * math.exp(self.learning_rate * (target_entropy - mean_entropy)),
            )
        return {
            "loss": sum(critic_losses) / len(critic_losses),
            "samples": float(len(critic_losses)),
            "alpha": self.alpha,
            "entropy": sum(entropies) / len(entropies),
        }

    def state_dict(self) -> dict:
        return {
            "algorithm": "sac",
            "network": self.network.state_dict(),
            "alpha": self.alpha,
        }

    def load_state_dict(self, state: dict) -> None:
        self.network.load_state_dict(state["network"])
        self.target.load_state_dict(state["network"])
        self.alpha = float(state.get("alpha", self.alpha))


__all__ = ["SACAgent"]
