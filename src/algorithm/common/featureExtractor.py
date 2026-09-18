"""PyTorch dual-resolution CNN and compact replay serialization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


def compress_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Pack binary rasters and copy small vectors for memory-efficient replay."""
    if observation.get("_compressed"):
        return observation
    global_weather = np.asarray(observation["global_weather"], dtype=np.uint8)
    local_weather = np.asarray(observation["local_weather"], dtype=np.uint8)
    action_mask = np.asarray(observation["action_mask"], dtype=np.uint8)
    guidance = np.clip(
        np.rint(np.asarray(observation["guidance_global"], dtype=np.float32) * 255.0),
        0,
        255,
    ).astype(np.uint8)
    return {
        "_compressed": True,
        "global_shape": global_weather.shape,
        "global_bits": np.packbits(global_weather.reshape(-1)).tobytes(),
        "local_shape": local_weather.shape,
        "local_bits": np.packbits(local_weather.reshape(-1)).tobytes(),
        "kinematics": np.asarray(observation["kinematics"], dtype=np.float32).copy(),
        "action_shape": action_mask.shape,
        "action_bits": np.packbits(action_mask.reshape(-1)).tobytes(),
        "guidance_shape": guidance.shape,
        "guidance_bytes": guidance.tobytes(),
        "guidance_vector": np.asarray(observation["guidance_vector"], dtype=np.float32).copy(),
    }


def decompress_observation(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    if not observation.get("_compressed"):
        return {
            "global_weather": np.asarray(observation["global_weather"], dtype=np.uint8),
            "local_weather": np.asarray(observation["local_weather"], dtype=np.uint8),
            "kinematics": np.asarray(observation["kinematics"], dtype=np.float32),
            "action_mask": np.asarray(observation["action_mask"], dtype=np.bool_),
            "guidance_global": np.asarray(observation["guidance_global"], dtype=np.float32),
            "guidance_vector": np.asarray(observation["guidance_vector"], dtype=np.float32),
        }

    def unpack(name: str, shape_name: str) -> np.ndarray:
        shape = tuple(observation[shape_name])
        size = int(np.prod(shape))
        packed = np.frombuffer(observation[name], dtype=np.uint8)
        return np.unpackbits(packed, count=size).reshape(shape).astype(np.uint8)

    guidance_shape = tuple(observation["guidance_shape"])
    guidance = (
        np.frombuffer(observation["guidance_bytes"], dtype=np.uint8)
        .reshape(guidance_shape)
        .astype(np.float32)
        / 255.0
    )
    return {
        "global_weather": unpack("global_bits", "global_shape"),
        "local_weather": unpack("local_bits", "local_shape"),
        "kinematics": np.asarray(observation["kinematics"], dtype=np.float32),
        "action_mask": unpack("action_bits", "action_shape").astype(np.bool_),
        "guidance_global": guidance,
        "guidance_vector": np.asarray(observation["guidance_vector"], dtype=np.float32),
    }


def observations_to_tensors(
    observations: Sequence[dict[str, Any]], device: torch.device
) -> dict[str, Tensor]:
    restored = [decompress_observation(observation) for observation in observations]
    return {
        "global_weather": torch.as_tensor(
            np.stack([item["global_weather"] for item in restored]),
            dtype=torch.float32,
            device=device,
        ),
        "local_weather": torch.as_tensor(
            np.stack([item["local_weather"] for item in restored]),
            dtype=torch.float32,
            device=device,
        ),
        "kinematics": torch.as_tensor(
            np.stack([item["kinematics"] for item in restored]),
            dtype=torch.float32,
            device=device,
        ),
        "action_mask": torch.as_tensor(
            np.stack([item["action_mask"] for item in restored]),
            dtype=torch.bool,
            device=device,
        ),
        "guidance_global": torch.as_tensor(
            np.stack([item["guidance_global"] for item in restored]),
            dtype=torch.float32,
            device=device,
        ),
        "guidance_vector": torch.as_tensor(
            np.stack([item["guidance_vector"] for item in restored]),
            dtype=torch.float32,
            device=device,
        ),
    }


class _MapEncoder(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.layers(values)


class DualResolutionFeatureExtractor(nn.Module):
    """Encode global history/guidance, local history and vehicle state."""

    output_dim = 32 * 4 * 4 * 2 + 64

    def __init__(self, history_frames: int) -> None:
        super().__init__()
        self.history_frames = history_frames
        self.global_encoder = _MapEncoder(history_frames + 1)
        self.local_encoder = _MapEncoder(history_frames)
        self.vector_encoder = nn.Sequential(nn.Linear(17, 64), nn.ReLU())

    def forward(self, observation: dict[str, Tensor]) -> Tensor:
        global_input = torch.cat(
            (observation["global_weather"], observation["guidance_global"].unsqueeze(1)),
            dim=1,
        )
        vector = torch.cat((observation["kinematics"], observation["guidance_vector"]), dim=1)
        return torch.cat(
            (
                self.global_encoder(global_input),
                self.local_encoder(observation["local_weather"]),
                self.vector_encoder(vector),
            ),
            dim=1,
        )

    @staticmethod
    def compact(observation: dict[str, Any]) -> dict[str, Any]:
        return compress_observation(observation)


__all__ = [
    "DualResolutionFeatureExtractor",
    "compress_observation",
    "decompress_observation",
    "observations_to_tensors",
]
