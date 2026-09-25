from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from dawn.sim.utils.env_wrappers import VecEnvWrapper


@dataclass(frozen=True)
class WorldModelObservationSpec:
    prop_with_command_dim: int

    depth_shape: tuple[int, int, int]
    forward_height_map_dim: int


def infer_world_model_observation_spec(obs_dict: dict[str, torch.Tensor]) -> WorldModelObservationSpec:
    prop_clean = _require_obs(obs_dict, "prop_clean")
    command = _require_obs(obs_dict, "command")
    depth = _require_obs(obs_dict, "depth")
    forward_height_map = _require_obs(obs_dict, "forward_height_map")
    return WorldModelObservationSpec(
        prop_with_command_dim=int(prop_clean.shape[-1] + command.shape[-1]),
        depth_shape=infer_depth_shape(depth),
        forward_height_map_dim=int(forward_height_map.shape[-1]),
    )


def infer_depth_shape(depth: torch.Tensor) -> tuple[int, int, int]:
    if depth.ndim != 4:
        raise ValueError("World model depth image must have shape (num_envs, C, H, W).")
    _, channels, height, width = depth.shape
    return int(channels), int(height), int(width)


def validate_world_model_observation_keys(obs_dict: dict[str, torch.Tensor], required_keys: tuple[str, ...]) -> None:
    missing_keys = [key for key in required_keys if key not in obs_dict]
    if missing_keys:
        missing_display = ", ".join(missing_keys)
        raise KeyError(f"World model requires observations: missing {missing_display}")


def build_world_model_obs_template(
    prop: torch.Tensor,
    command: torch.Tensor,
    wm_is_first: torch.Tensor,
    device: torch.device | str,
    depth_shape: tuple[int, int, int],
) -> dict[str, torch.Tensor]:
    return {
        "prop": torch.cat([prop, command], dim=-1).to(device),
        "is_first": wm_is_first,
        "image": torch.zeros((prop.shape[0], *depth_shape), device=device),
    }


def infer_world_model_depth_index(env: VecEnvWrapper) -> np.ndarray:
    depth_cfg = getattr(env.cfg, "depth", None)
    if depth_cfg is None:
        raise ValueError("World model requires env.cfg.depth.")

    uses_camera = bool(getattr(depth_cfg, "use_camera", False))
    if not uses_camera:
        raise ValueError("World model requires env.cfg.depth.use_camera=True.")
    depth_index = np.asarray(getattr(env, "depth_index", np.arange(env.num_envs)), dtype=np.int64)
    _validate_depth_index(num_envs=env.num_envs, depth_index=depth_index)
    return depth_index


def _validate_depth_index(num_envs: int, depth_index: np.ndarray) -> None:
    if depth_index.ndim != 1:
        raise ValueError("World model requires depth_index to be a 1D array.")
    if np.any(depth_index < 0) or np.any(depth_index >= num_envs):
        raise ValueError("World model requires depth_index values within [0, num_envs).")
    if np.unique(depth_index).shape[0] != depth_index.shape[0]:
        raise ValueError("World model requires depth_index to contain unique environment ids.")


def _require_obs(obs_dict: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    value = obs_dict.get(key)
    if value is None:
        raise KeyError(f"World model requires '{key}' in observations.")
    return value
