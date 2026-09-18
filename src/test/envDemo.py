"""Run a short random-action environment demonstration."""

from __future__ import annotations

import random
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from env import ReturnEnv  # noqa: E402


def main() -> None:
    env = ReturnEnv()
    observation, _ = env.reset(seed=0)
    rng = random.Random(0)
    for step in range(20):
        valid = [index for index, enabled in enumerate(observation["action_mask"]) if enabled]
        action = rng.choice(valid)
        observation, reward, terminated, truncated, info = env.step(action)
        print(step + 1, action, round(reward, 3), info["helicopter_position_nm"])
        if terminated or truncated:
            print("outcome:", info["outcome"])
            break


if __name__ == "__main__":
    main()
