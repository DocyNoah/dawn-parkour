from typing import TYPE_CHECKING

import torch
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.assets.rigid_object.rigid_object import RigidObject


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lin_vel_clip: float = 0.1,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    lin_vel = asset.data.root_lin_vel_b[:, :2].clone()
    command_vel = env.command_manager.get_command(command_name)[:, :2].clone()
    lin_vel_upper_bound = torch.where(command_vel < 0, 1e5, command_vel + lin_vel_clip)
    lin_vel_lower_bound = torch.where(command_vel > 0, -1e5, command_vel - lin_vel_clip)
    clip_lin_vel = torch.clip(lin_vel, min=lin_vel_lower_bound, max=lin_vel_upper_bound)
    lin_vel_error = torch.sum(torch.square(command_vel - clip_lin_vel), dim=1)
    return torch.exp(-lin_vel_error / std)


def stuck(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    asset: RigidObject = env.scene["robot"]
    base_lin_vel_x = asset.data.root_lin_vel_b[:, 0]
    command_x = env.command_manager.get_command(command_name)[:, 0]

    return (torch.abs(base_lin_vel_x) < 0.1) * (torch.abs(command_x) > 0.1).float()


def heading_deviation(env: ManagerBasedRLEnv) -> torch.Tensor:
    asset: RigidObject = env.scene["robot"]

    forward_vec = torch.tensor([1.0, 0.0, 0.0], device=env.device, dtype=torch.float32).repeat((env.num_envs, 1))
    forward = quat_apply(asset.data.root_quat_w, forward_vec)
    heading = torch.atan2(forward[:, 1], forward[:, 0])

    heading_deviated = (heading > 1.0) | (heading < -1.0)
    return heading_deviated
