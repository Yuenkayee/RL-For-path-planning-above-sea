"""Canonical experiment groups used by training and evaluation scripts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentGroup:
    name: str
    family: str
    role: str
    trainable: bool
    config_file: str


EXPERIMENT_GROUPS = (
    ExperimentGroup(
        "global_guided_masked_ppo",
        "reinforcement_learning",
        "main",
        True,
        "config/ppoConfig.json",
    ),
    ExperimentGroup(
        "global_guided_masked_dueling_double_dqn",
        "reinforcement_learning",
        "rl_baseline",
        True,
        "config/dqnConfig.json",
    ),
    ExperimentGroup(
        "discrete_maximum_entropy_sac",
        "reinforcement_learning",
        "optional_rl_baseline",
        True,
        "config/sacConfig.json",
    ),
    ExperimentGroup(
        "rolling_horizon_sipp",
        "classical_planning",
        "classical_baseline",
        False,
        "config/plannerConfig.json",
    ),
    ExperimentGroup(
        "rolling_time_expanded_astar",
        "classical_planning",
        "classical_baseline",
        False,
        "config/plannerConfig.json",
    ),
)
