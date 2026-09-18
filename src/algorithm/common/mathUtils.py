"""Small numerical helpers used to avoid a mandatory NumPy dependency."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def dot(first: Sequence[float], second: Sequence[float]) -> float:
    return sum(left * right for left, right in zip(first, second, strict=True))


def masked_softmax(
    logits: Sequence[float], mask: Sequence[bool], temperature: float = 1.0
) -> tuple[float, ...]:
    if len(logits) != len(mask):
        raise ValueError("logits and mask must have identical lengths")
    valid = [index for index, enabled in enumerate(mask) if enabled]
    if not valid:
        raise ValueError("at least one action must be valid")
    maximum = max(logits[index] / temperature for index in valid)
    weights = [0.0] * len(logits)
    for index in valid:
        weights[index] = math.exp(logits[index] / temperature - maximum)
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def sample_categorical(probabilities: Sequence[float], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return index
    return len(probabilities) - 1


def argmax_masked(values: Sequence[float], mask: Sequence[bool]) -> int:
    valid = [index for index, enabled in enumerate(mask) if enabled]
    if not valid:
        raise ValueError("at least one action must be valid")
    return max(valid, key=lambda index: values[index])


__all__ = ["argmax_masked", "dot", "masked_softmax", "sample_categorical"]
