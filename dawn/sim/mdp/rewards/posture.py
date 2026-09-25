from typing import TYPE_CHECKING

import torch
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

from dawn.sim.mdp.rewards.utils import _get_foot_contact_state

if TYPE_CHECKING:
    from isaaclab.assets.articulation import Articulation


def joint_deviation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(angle), dim=1)


def stand_still_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    contact_threshold: float = 1.0,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    cmd_lin_vel = torch.norm(command[:, :2], dim=1)
    cmd_ang_vel = torch.abs(command[:, 2])
    is_zero_command = (cmd_lin_vel < command_threshold) & (cmd_ang_vel < command_threshold)

    feet_in_contact = _get_foot_contact_state(env, sensor_cfg, contact_threshold)

    four_feet_not_contact = torch.sum((~feet_in_contact).float(), dim=1) > 0

    return four_feet_not_contact.float() * is_zero_command.float()


def stand_still_default_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.1,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    cmd_lin_vel = torch.norm(command[:, :2], dim=1)
    cmd_ang_vel = torch.abs(command[:, 2])
    is_zero_command = (cmd_lin_vel < command_threshold) & (cmd_ang_vel < command_threshold)

    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(angle), dim=1) * is_zero_command.float()
