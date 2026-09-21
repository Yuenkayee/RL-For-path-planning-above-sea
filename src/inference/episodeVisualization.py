"""Record and visualize one deterministic planning episode."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(_REPOSITORY_ROOT / "build" / ".matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from env.returnEnv import ReturnEnv  # noqa: E402
from model.weatherMap import WeatherMapSnapshot  # noqa: E402


@dataclass(frozen=True)
class EpisodeFrame:
    """One captured state in an episode animation."""

    step: int
    elapsed_seconds: float
    snapshot: WeatherMapSnapshot
    action: int | None
    step_reward: float
    cumulative_reward: float


@dataclass(frozen=True)
class EpisodeTrace:
    """Complete paths plus sampled weather frames for one episode."""

    method: str
    seed: int
    outcome: str
    total_reward: float
    steps: int
    helicopter_path_nm: tuple[tuple[float, float], ...]
    frigate_path_nm: tuple[tuple[float, float], ...]
    frames: tuple[EpisodeFrame, ...]


def run_episode_trace(
    env: ReturnEnv,
    policy: Any,
    *,
    method: str,
    seed: int,
    max_steps: int | None = None,
    frame_stride: int = 10,
    reset_options: dict[str, Any] | None = None,
) -> EpisodeTrace:
    """Run one deterministic episode and retain snapshots for visualization."""
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")

    observation, _ = env.reset(seed=seed, options=reset_options)
    helicopter_path = [(env.helicopter.x_nm, env.helicopter.y_nm)]
    frigate_path = [(env.frigate.x_nm, env.frigate.y_nm)]
    frames = [
        EpisodeFrame(
            step=0,
            elapsed_seconds=0.0,
            snapshot=env.weather_map.snapshot(),
            action=None,
            step_reward=0.0,
            cumulative_reward=0.0,
        )
    ]
    total_reward = 0.0
    steps = 0

    while True:
        action, _ = policy.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
        total_reward += reward
        helicopter_path.append(info["helicopter_position_nm"])
        frigate_path.append(info["frigate_position_nm"])

        reached_limit = max_steps is not None and steps >= max_steps
        finished = terminated or truncated or reached_limit
        if steps % frame_stride == 0 or finished:
            frames.append(
                EpisodeFrame(
                    step=steps,
                    elapsed_seconds=info["elapsed_seconds"],
                    snapshot=env.weather_map.snapshot(),
                    action=action,
                    step_reward=reward,
                    cumulative_reward=total_reward,
                )
            )

        if finished:
            return EpisodeTrace(
                method=method,
                seed=seed,
                outcome=info.get("outcome") or "step_limit",
                total_reward=total_reward,
                steps=steps,
                helicopter_path_nm=tuple(helicopter_path),
                frigate_path_nm=tuple(frigate_path),
                frames=tuple(frames),
            )


def _draw_frame(axis: Any, trace: EpisodeTrace, frame: EpisodeFrame) -> None:
    snapshot = frame.snapshot
    map_width, map_height = snapshot.area_size_nm
    axis.clear()

    weather_colors = ListedColormap(("#b9dfe8", "#c83e4d"))
    axis.imshow(
        np.asarray(snapshot.global_grid, dtype=np.uint8),
        origin="lower",
        extent=(0.0, map_width, 0.0, map_height),
        interpolation="nearest",
        cmap=weather_colors,
        vmin=0,
        vmax=1,
        zorder=0,
    )
    local_x, local_y = snapshot.local_origin_nm
    local_width, local_height = snapshot.local_size_nm
    axis.imshow(
        np.asarray(snapshot.local_grid, dtype=np.uint8),
        origin="lower",
        extent=(local_x, local_x + local_width, local_y, local_y + local_height),
        interpolation="nearest",
        cmap=weather_colors,
        vmin=0,
        vmax=1,
        zorder=1,
    )
    axis.add_patch(
        Rectangle(
            (local_x, local_y),
            local_width,
            local_height,
            fill=False,
            edgecolor="#1b9e77",
            linewidth=1.5,
            zorder=2,
            label="Local weather window",
        )
    )

    path_end = frame.step + 1
    helicopter = trace.helicopter_path_nm[:path_end]
    frigate = trace.frigate_path_nm[:path_end]
    helicopter_x, helicopter_y = zip(*helicopter, strict=True)
    frigate_x, frigate_y = zip(*frigate, strict=True)
    axis.plot(
        helicopter_x,
        helicopter_y,
        color="#ffb000",
        linewidth=2.4,
        label="Helicopter path",
        zorder=4,
    )
    axis.plot(
        frigate_x,
        frigate_y,
        color="#2f4858",
        linestyle="--",
        linewidth=1.8,
        label="Frigate path",
        zorder=3,
    )
    axis.scatter(
        helicopter_x[-1],
        helicopter_y[-1],
        marker="^",
        s=90,
        color="#ffe066",
        edgecolor="black",
        label="Helicopter",
        zorder=6,
    )
    axis.scatter(
        frigate_x[-1],
        frigate_y[-1],
        marker="s",
        s=75,
        color="#546e7a",
        edgecolor="white",
        label="Frigate",
        zorder=5,
    )
    axis.scatter(
        helicopter_x[0],
        helicopter_y[0],
        marker="o",
        s=35,
        facecolor="white",
        edgecolor="black",
        label="Start",
        zorder=5,
    )

    action_text = "initial" if frame.action is None else str(frame.action)
    title = (
        f"{trace.method.upper()} | seed={trace.seed} | step={frame.step}/{trace.steps} | "
        f"t={frame.elapsed_seconds / 60.0:.1f} min\n"
        f"action={action_text} | step reward={frame.step_reward:.3f} | "
        f"total reward={frame.cumulative_reward:.3f}"
    )
    if frame.step == trace.steps:
        title += f" | outcome={trace.outcome}"
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("East / nautical miles")
    axis.set_ylabel("North / nautical miles")
    axis.set_xlim(0.0, map_width)
    axis.set_ylim(0.0, map_height)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="white", alpha=0.22, linewidth=0.5)
    axis.legend(loc="upper right", fontsize=7, framealpha=0.9)


def save_episode_visualization(
    trace: EpisodeTrace,
    path: str | Path,
    *,
    fps: int = 8,
    dpi: int = 120,
) -> Path:
    """Save an episode as an animated GIF or a final-state PNG."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    output = Path(path).expanduser().resolve()
    if output.suffix.lower() not in {".gif", ".png"}:
        raise ValueError("visualization output must end in .gif or .png")
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 8))
    try:
        if output.suffix.lower() == ".png":
            _draw_frame(axis, trace, trace.frames[-1])
            figure.tight_layout()
            figure.savefig(output, dpi=dpi)
        else:
            def update(frame_index: int) -> None:
                _draw_frame(axis, trace, trace.frames[frame_index])
                figure.tight_layout()

            animation = FuncAnimation(
                figure,
                update,
                frames=len(trace.frames),
                interval=1000 / fps,
                repeat=False,
            )
            animation.save(output, writer=PillowWriter(fps=fps), dpi=dpi)
    finally:
        plt.close(figure)
    return output


__all__ = [
    "EpisodeFrame",
    "EpisodeTrace",
    "run_episode_trace",
    "save_episode_visualization",
]
