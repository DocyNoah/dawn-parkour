from typing import TYPE_CHECKING

import torch
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

from dawn.sim.mdp.rewards.rewards import _get_terrain_type_mask

if TYPE_CHECKING:
    from isaaclab.assets.rigid_object.rigid_object import RigidObject


def off_center_termination(
    env: ManagerBasedRLEnv, max_distance: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    robot_y = asset.data.root_pos_w[:, 1]

    terrain_center_y = env.scene.env_origins[:, 1]

    y_distance = torch.abs(robot_y - terrain_center_y)

    is_too_far = y_distance > max_distance

    return is_too_far


def no_progress(
    env: ManagerBasedRLEnv,
    max_stagnation_steps: int = 100,
    command_name: str = "base_velocity",
    min_command_speed: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    current_x = asset.data.root_pos_w[:, 0]

    commanded_vel = env.command_manager.get_command(command_name)[:, :2]
    cmd_speed = torch.norm(commanded_vel, dim=1)
    is_standing = cmd_speed <= min_command_speed

    if "max_x_position" not in env.extras:
        env.extras["max_x_position"] = current_x.clone()
        env.extras["steps_since_progress"] = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    reset_mask = env.episode_length_buf <= 1
    env.extras["max_x_position"][reset_mask] = current_x[reset_mask]
    env.extras["steps_since_progress"][reset_mask] = 0

    progress_made = current_x > env.extras["max_x_position"]

    env.extras["max_x_position"] = torch.maximum(env.extras["max_x_position"], current_x)
    env.extras["steps_since_progress"] = torch.where(
        progress_made | is_standing,
        torch.zeros_like(env.extras["steps_since_progress"]),
        env.extras["steps_since_progress"] + 1,
    )

    should_terminate = env.extras["steps_since_progress"] >= max_stagnation_steps

    return should_terminate


def root_height_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_terrain_type: bool = False,
    terrain_type_names_list: list[str] | None = None,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    is_below = asset.data.root_pos_w[:, 2] < minimum_height

    max_level = env.scene.terrain.max_terrain_level - 1
    is_max_level = env.scene.terrain.terrain_levels == max_level

    robot_x = asset.data.root_pos_w[:, 0]
    terrain_center_x = env.scene.env_origins[:, 0]

    clear = robot_x > terrain_center_x + 3.9

    is_max_level_clear = is_max_level & clear
    is_below = is_below & ~is_max_level_clear

    if use_terrain_type:
        terrain_type_mask = _get_terrain_type_mask(
            terrain=env.scene.terrain,
            num_envs=env.num_envs,
            device=env.device,
            terrain_type_names_list=terrain_type_names_list,
        )
        return is_below & terrain_type_mask

    return is_below


def is_max_level_clear(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    max_level = env.scene.terrain.max_terrain_level - 1
    is_max_level = env.scene.terrain.terrain_levels == max_level

    robot_x = asset.data.root_pos_w[:, 0]
    terrain_center_x = env.scene.env_origins[:, 0]

    clear = robot_x > terrain_center_x + 5

    is_max_level_clear = is_max_level & clear

    return is_max_level_clear
