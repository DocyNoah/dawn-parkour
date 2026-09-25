from typing import TYPE_CHECKING

import torch
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

from dawn.sim.mdp.rewards.utils import (
    _get_foot_contact_state,
    _get_terrain_edge_map,
    _get_terrain_type_mask,
    _world_to_pixel_coordinates,
)

if TYPE_CHECKING:
    from isaaclab.assets.articulation import Articulation
    from isaaclab.assets.rigid_object.rigid_object import RigidObject
    from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor

    from dawn.sim.mdp.terrains.terrain_importer import TerrainImporterWithEdges


def stumble(
    env: ManagerBasedRLEnv,
    ratio_threshold: float,
    sensor_cfg: SceneEntityCfg,
    use_terrain_type: bool = False,
    terrain_type_names_list: list[str] | None = None,
) -> torch.Tensor:
    if use_terrain_type:
        terrain_type_mask = _get_terrain_type_mask(
            env.scene.terrain,
            env.scene.num_envs,
            env.device,
            terrain_type_names_list,
        )
    else:
        terrain_type_mask = torch.ones(env.scene.num_envs, device=env.device, dtype=torch.bool)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history

    contact_xy = torch.norm(net_contact_forces[:, 0, sensor_cfg.body_ids, :2], dim=2)
    contact_z = net_contact_forces[:, 0, sensor_cfg.body_ids, 2]

    stumble = torch.any(contact_xy > (ratio_threshold * contact_z), dim=1)

    stumble = stumble.float()

    return stumble * terrain_type_mask


def edge_avoidance(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    min_terrain_level: int = 3,
    use_terrain_type: bool = False,
    terrain_type_names_list: list[str] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
) -> torch.Tensor:
    terrain: TerrainImporterWithEdges = env.scene.terrain
    edge_x_tensor = _get_terrain_edge_map(terrain, env.device)

    if use_terrain_type:
        terrain_type_mask = _get_terrain_type_mask(
            terrain=terrain,
            num_envs=env.num_envs,
            device=env.device,
            terrain_type_names_list=terrain_type_names_list,
        )
    else:
        terrain_type_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

    if edge_x_tensor is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    robot: Articulation = env.scene.articulations[asset_cfg.name]
    foot_positions_w = robot.data.body_pos_w[:, sensor_cfg.body_ids, :2]

    horizontal_scale = terrain.cfg.terrain_generator.horizontal_scale
    foot_pos_pixels = _world_to_pixel_coordinates(foot_positions_w, terrain, horizontal_scale, edge_x_tensor.shape)

    is_in_contact = _get_foot_contact_state(env, sensor_cfg, threshold)

    feet_on_edges = edge_x_tensor[foot_pos_pixels[..., 0], foot_pos_pixels[..., 1]]

    feet_edge_penalty = is_in_contact & feet_on_edges

    total_penalty = torch.sum(feet_edge_penalty.float(), dim=-1)

    terrain_level_mask = terrain.terrain_levels >= min_terrain_level

    total_penalty = total_penalty * terrain_level_mask.float() * terrain_type_mask.float()
    return total_penalty * 3.0


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.norm(net_contact_forces[:, 0, sensor_cfg.body_ids, :], dim=-1) > threshold

    return torch.sum(1.0 * (is_contact), dim=1)


def off_center_penalty(
    env: ManagerBasedRLEnv,
    safe_distance: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    robot_y = asset.data.root_pos_w[:, 1]
    terrain_center_y = env.scene.env_origins[:, 1]
    y_distance = torch.abs(robot_y - terrain_center_y)

    excess_cm = torch.clamp(y_distance - safe_distance, min=0.0) * 100.0
    return torch.square(excess_cm)
