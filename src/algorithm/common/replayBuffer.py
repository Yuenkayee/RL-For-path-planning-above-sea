"""Replay and rollout buffers shared by learning algorithms."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Transition:
    observation: dict[str, Any]
    action: int
    reward: float
    next_observation: dict[str, Any]
    done: bool


class ReplayBuffer:
    def __init__(
        self, capacity: int, *, prioritized: bool = False, seed: int | None = None
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.prioritized = prioritized
        self._rng = random.Random(seed)
        self._items: list[Transition] = []
        self._priorities: list[float] = []
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._items)

    def add(self, transition: Transition, priority: float | None = None) -> None:
        selected_priority = max(1e-6, float(priority or max(self._priorities, default=1.0)))
        if len(self._items) < self.capacity:
            self._items.append(transition)
            self._priorities.append(selected_priority)
        else:
            self._items[self._cursor] = transition
            self._priorities[self._cursor] = selected_priority
        self._cursor = (self._cursor + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple[tuple[int, Transition], ...]:
        if batch_size <= 0 or batch_size > len(self._items):
            raise ValueError("batch_size must be positive and no larger than the buffer")
        if self.prioritized:
            indices = self._rng.choices(
                range(len(self._items)), weights=self._priorities, k=batch_size
            )
        else:
            indices = self._rng.sample(range(len(self._items)), batch_size)
        return tuple((index, self._items[index]) for index in indices)

    def update_priority(self, index: int, priority: float) -> None:
        self._priorities[index] = max(1e-6, float(priority))


@dataclass
class RolloutStep:
    observation: dict[str, Any]
    action: int
    reward: float
    value: float
    log_probability: float
    done: bool
    environment_id: int = 0


class RolloutBuffer:
    def __init__(self) -> None:
        self.steps: list[RolloutStep] = []

    def add(self, step: RolloutStep) -> None:
        self.steps.append(step)

    def clear(self) -> None:
        self.steps.clear()

    def __len__(self) -> int:
        return len(self.steps)


__all__ = ["ReplayBuffer", "RolloutBuffer", "RolloutStep", "Transition"]
