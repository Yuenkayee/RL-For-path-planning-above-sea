"""Double DQN agent with prioritized replay and n-step returns."""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from algorithm.common.featureExtractor import (
    compress_observation,
    observations_to_tensors,
)
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
        device: str = "cpu",
    ) -> None:
        if seed is not None:
            torch.manual_seed(seed)
        self.device = torch.device(device)
        history_frames = int(np.asarray(sample_observation["global_weather"]).shape[0])
        self.online = DuelingQNetwork(history_frames, action_count).to(self.device)
        self.target = DuelingQNetwork(history_frames, action_count).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
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
        tensors = observations_to_tensors((observation,), self.device)
        self.online.eval()
        with torch.no_grad():
            q_values = self.online(tensors)[0]
        valid = [index for index, enabled in enumerate(observation["action_mask"]) if enabled]
        if not deterministic and self._rng.random() < self.epsilon:
            action = self._rng.choice(valid)
        else:
            masked = q_values.masked_fill(~tensors["action_mask"][0], -torch.inf)
            action = int(torch.argmax(masked).item())
        return action, {"q_value": float(q_values[action].item())}

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
            compress_observation(transition.observation),
            transition.action,
            transition.reward,
            compress_observation(transition.next_observation),
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
        sampled = self.replay.sample(self.batch_size)
        indices = [item[0] for item in sampled]
        transitions = [item[1] for item in sampled]
        observations = observations_to_tensors(
            [transition.observation for transition in transitions], self.device
        )
        next_observations = observations_to_tensors(
            [transition.next_observation for transition in transitions], self.device
        )
        actions = torch.as_tensor(
            [transition.action for transition in transitions], dtype=torch.long, device=self.device
        )
        rewards = torch.as_tensor(
            [transition.reward for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(
            [transition.done for transition in transitions], dtype=torch.float32, device=self.device
        )
        self.online.train()
        predictions = self.online(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            online_next = self.online(next_observations).masked_fill(
                ~next_observations["action_mask"], -torch.inf
            )
            next_actions = online_next.argmax(dim=1)
            target_next = (
                self.target(next_observations).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            )
            targets = rewards + (1.0 - dones) * self.discount_factor**self.n_step * target_next
        td_errors = targets - predictions
        loss = F.smooth_l1_loss(predictions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        for index, error in zip(indices, td_errors.detach().abs().cpu().tolist(), strict=True):
            self.replay.update_priority(index, error)
        self.update_count += 1
        if self.update_count % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return {
            "loss": float(loss.item()),
            "samples": float(len(transitions)),
        }

    def state_dict(self) -> dict:
        return {
            "algorithm": "dqn",
            "network": self.online.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.online.load_state_dict(state["network"])
        self.target.load_state_dict(self.online.state_dict())
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self.update_count = int(state.get("update_count", 0))


__all__ = ["DQNAgent"]
