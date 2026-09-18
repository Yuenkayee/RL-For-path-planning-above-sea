"""Discrete maximum-entropy Soft Actor-Critic comparison agent."""

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
        device: str = "cpu",
    ) -> None:
        if seed is not None:
            torch.manual_seed(seed)
        self.device = torch.device(device)
        history_frames = int(np.asarray(sample_observation["global_weather"]).shape[0])
        self.network = SACNetwork(history_frames, action_count).to(self.device)
        self.target = SACNetwork(history_frames, action_count).to(self.device)
        self.target.load_state_dict(self.network.state_dict())
        self.target.eval()
        self.actor_optimizer = torch.optim.Adam(self.network.actor_parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.network.critic_parameters(), lr=learning_rate)
        self.replay = ReplayBuffer(replay_capacity, seed=seed)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.soft_update_factor = soft_update_factor
        self.batch_size = batch_size
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.log_alpha = torch.tensor(
            np.log(entropy_temperature), dtype=torch.float32, device=self.device
        )
        self.log_alpha.requires_grad_(automatic_entropy_tuning)
        self.alpha_optimizer = (
            torch.optim.Adam((self.log_alpha,), lr=learning_rate)
            if automatic_entropy_tuning
            else None
        )
        self.target_entropy = 0.8 * np.log(action_count)

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().detach().item())

    def predict(
        self, observation: dict[str, Any], *, deterministic: bool = False
    ) -> tuple[int, dict[str, float]]:
        tensors = observations_to_tensors((observation,), self.device)
        self.network.eval()
        with torch.no_grad():
            logits = self.network.actor_logits(tensors)
            distribution = Categorical(logits=logits)
            actions = torch.argmax(logits, dim=1) if deterministic else distribution.sample()
            probabilities = distribution.probs
        action = int(actions[0].item())
        return action, {"probability": float(probabilities[0, action].item())}

    def observe(self, transition: Transition) -> None:
        self.replay.add(
            Transition(
                compress_observation(transition.observation),
                transition.action,
                transition.reward,
                compress_observation(transition.next_observation),
                transition.done,
            )
        )

    def update(self) -> dict[str, float]:
        if len(self.replay) < self.batch_size:
            return {"loss": 0.0, "samples": 0.0, "alpha": self.alpha}
        transitions = [item[1] for item in self.replay.sample(self.batch_size)]
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

        self.network.train()
        with torch.no_grad():
            next_logits = self.network.actor_logits(next_observations)
            next_log_probabilities = torch.log_softmax(next_logits, dim=1)
            next_probabilities = torch.softmax(next_logits, dim=1)
            target_q1, target_q2 = self.target.critics(next_observations)
            target_minimum = torch.minimum(target_q1, target_q2)
            soft_value = (
                next_probabilities
                * (target_minimum - self.log_alpha.exp() * next_log_probabilities)
            ).sum(dim=1)
            targets = rewards + (1.0 - dones) * self.discount_factor * soft_value

        q1, q2 = self.network.critics(observations)
        q1_selected = q1.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_selected = q2.gather(1, actions.unsqueeze(1)).squeeze(1)
        critic_loss = F.mse_loss(q1_selected, targets) + F.mse_loss(q2_selected, targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        logits = self.network.actor_logits(observations)
        log_probabilities = torch.log_softmax(logits, dim=1)
        probabilities = torch.softmax(logits, dim=1)
        with torch.no_grad():
            q1_policy, q2_policy = self.network.critics(observations)
            minimum_q = torch.minimum(q1_policy, q2_policy)
        actor_loss = (
            (probabilities * (self.log_alpha.exp().detach() * log_probabilities - minimum_q))
            .sum(dim=1)
            .mean()
        )
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        entropy = -(probabilities * log_probabilities).sum(dim=1).mean()
        if self.alpha_optimizer is not None:
            alpha_loss = self.log_alpha * (entropy.detach() - self.target_entropy)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        tau = self.soft_update_factor
        with torch.no_grad():
            for target_parameter, source_parameter in zip(
                self.target.parameters(), self.network.parameters(), strict=True
            ):
                target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)
        return {
            "loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "samples": float(len(transitions)),
            "alpha": self.alpha,
            "entropy": float(entropy.item()),
        }

    def state_dict(self) -> dict:
        return {
            "algorithm": "sac",
            "network": self.network.state_dict(),
            "target": self.target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.network.load_state_dict(state["network"])
        self.target.load_state_dict(state.get("target", state["network"]))
        if "actor_optimizer" in state:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        if "critic_optimizer" in state:
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if "log_alpha" in state:
            self.log_alpha.data.copy_(state["log_alpha"].to(self.device))


__all__ = ["SACAgent"]
