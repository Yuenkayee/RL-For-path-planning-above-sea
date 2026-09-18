"""Double DQN agent with prioritized replay and n-step returns."""

from __future__ import annotations

from collections import deque
import random
from typing import Any

from algorithm.common.featureExtractor import DualResolutionFeatureExtractor
from algorithm.common.mathUtils import argmax_masked
from algorithm.common.replayBuffer import ReplayBuffer, Transition

from .dqnNetwork import DuelingQNetwork


class DQNAgent:
    def __init__(
        self,
        action_count: int,
        sample_observation: dict[str, Any],
        *,
        learning_rate: float = 1e-4,
        discount_factor: float = 0.99,
        replay_capacity: int = 200000,
        batch_size: int = 128,
        n_step: int = 3,
        target_update_interval: int = 1000,
        prioritized_replay: bool = True,
        epsilon: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.extractor = DualResolutionFeatureExtractor()
        feature_count = len(self.extractor(sample_observation))
        self.online = DuelingQNetwork(feature_count, action_count, seed=seed)
        self.target = DuelingQNetwork(feature_count, action_count, seed=seed)
        self.target.copy_from(self.online)
        self.replay = ReplayBuffer(replay_capacity, prioritized=prioritized_replay, seed=seed)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.n_step = n_step
        self.target_update_interval = target_update_interval
        self.epsilon = epsilon
        self._rng = random.Random(seed)
        self._n_step_queue: deque[Transition] = deque()
        self.update_count = 0

    def predict(
        self, observation: dict[str, Any], *, deterministic: bool = False
    ) -> tuple[int, dict[str, float]]:
        features = self.extractor(observation)
        q_values = self.online.q_values(features)
        valid = [index for index, enabled in enumerate(observation["action_mask"]) if enabled]
        if not deterministic and self._rng.random() < self.epsilon:
            action = self._rng.choice(valid)
        else:
            action = argmax_masked(q_values, observation["action_mask"])
        return action, {"q_value": q_values[action]}

    def _emit_n_step(self) -> None:
        if not self._n_step_queue:
            return
        reward = 0.0
        final = self._n_step_queue[0]
        for index, transition in enumerate(self._n_step_queue):
            reward += self.discount_factor**index * transition.reward
            final = transition
            if transition.done or index + 1 >= self.n_step:
                break
        first = self._n_step_queue[0]
        self.replay.add(
            Transition(
                first.observation,
                first.action,
                reward,
                final.next_observation,
                final.done,
            )
        )
        self._n_step_queue.popleft()

    def observe(self, transition: Transition) -> None:
        compact = Transition(
            self.extractor.compact(transition.observation),
            transition.action,
            transition.reward,
            self.extractor.compact(transition.next_observation),
            transition.done,
        )
        self._n_step_queue.append(compact)
        if len(self._n_step_queue) >= self.n_step:
            self._emit_n_step()
        if transition.done:
            while self._n_step_queue:
                self._emit_n_step()

    def update(self) -> dict[str, float]:
        if len(self.replay) < self.batch_size:
            return {"loss": 0.0, "samples": 0.0}
        errors = []
        for index, transition in self.replay.sample(self.batch_size):
            features = self.extractor(transition.observation)
            if transition.done:
                target_value = transition.reward
            else:
                next_features = self.extractor(transition.next_observation)
                next_action = argmax_masked(
                    self.online.q_values(next_features),
                    transition.next_observation["action_mask"],
                )
                target_value = transition.reward + self.discount_factor**self.n_step * self.target.q_values(
                    next_features
                )[next_action]
            error = self.online.update(
                features, transition.action, target_value, self.learning_rate
            )
            self.replay.update_priority(index, abs(error))
            errors.append(error)
        self.update_count += 1
        if self.update_count % self.target_update_interval == 0:
            self.target.copy_from(self.online)
        return {
            "loss": sum(error * error for error in errors) / len(errors),
            "samples": float(len(errors)),
        }

    def state_dict(self) -> dict:
        return {
            "algorithm": "dqn",
            "network": self.online.state_dict(),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.online.load_state_dict(state["network"])
        self.target.copy_from(self.online)
        self.update_count = int(state.get("update_count", 0))


__all__ = ["DQNAgent"]
