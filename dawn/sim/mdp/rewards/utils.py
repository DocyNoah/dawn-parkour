from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

from dawn.sim.mdp.terrains.terrain_importer import TerrainImporterWithEdges

if TYPE_CHECKING:
    from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor


def _get_terrain_edge_map(terrain: TerrainImporterWithEdges, device: torch.device) -> torch.Tensor | None:
    if not terrain.has_edge_info:
        return None

    edge_x_map = terrain.global_terrain_edges_x
    if edge_x_map is None:
        return None

    return torch.from_numpy(edge_x_map).to(device=device, dtype=torch.bool)


def _get_terrain_type_mask(
    terrain: TerrainImporterWithEdges,
    num_envs: int,
    device: torch.device,
    terrain_type_names_list: list[str] | None = None,
) -> torch.Tensor:
    terrain_type_mask = torch.ones(num_envs, device=device, dtype=torch.bool)

    if not terrain_type_names_list:
        return terrain_type_mask

    if not hasattr(terrain, "terrain_type_names") or terrain.terrain_type_names is None:
        return terrain_type_mask

    terrain_types = terrain.terrain_types
    terrain_type_names = terrain.terrain_type_names

    terrain_types_np = terrain_types.cpu().numpy()
    if isinstance(terrain_type_names, np.ndarray):
        all_terrain_names = np.array(terrain_type_names)[terrain_types_np]
    else:
        all_terrain_names = np.array([terrain_type_names[i] for i in terrain_types_np])

    mask = np.zeros(len(all_terrain_names), dtype=bool)
    for i, terrain_name in enumerate(all_terrain_names):
        for filter_name in terrain_type_names_list:
            if filter_name in terrain_name:
                mask[i] = True
                break

    terrain_type_mask = torch.tensor(mask, device=device, dtype=torch.bool)

    return terrain_type_mask


def _get_foot_contact_state(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history

    current_forces = net_contact_forces[:, 0, sensor_cfg.body_ids, 2]
    previous_forces = net_contact_forces[:, 1, sensor_cfg.body_ids, 2]

    current_contact = current_forces > threshold
    previous_contact = previous_forces > threshold

    return torch.logical_or(current_contact, previous_contact)


def _world_to_pixel_coordinates(
    foot_positions_w: torch.Tensor,
    terrain: TerrainImporterWithEdges,
    horizontal_scale: float,
    edge_map_shape: tuple[int, int],
) -> torch.Tensor:
    device = foot_positions_w.device

    if hasattr(terrain.cfg, "terrain_generator") and terrain.cfg.terrain_generator is not None:
        terrain_size = terrain.cfg.terrain_generator.size
        num_rows = terrain.cfg.terrain_generator.num_rows
        num_cols = terrain.cfg.terrain_generator.num_cols

        total_width = num_rows * terrain_size[0]
        total_height = num_cols * terrain_size[1]
        offset_x = total_width / 2.0
        offset_y = total_height / 2.0
    else:
        offset_x = 0.0
        offset_y = 0.0

    offset_tensor = torch.tensor([offset_x, offset_y], device=device)
    foot_pos_pixels = (foot_positions_w + offset_tensor) / horizontal_scale
    foot_pos_pixels = foot_pos_pixels.round().long()

    foot_pos_pixels[..., 0] = torch.clamp(foot_pos_pixels[..., 0], 0, edge_map_shape[0] - 1)
    foot_pos_pixels[..., 1] = torch.clamp(foot_pos_pixels[..., 1], 0, edge_map_shape[1] - 1)

    return foot_pos_pixels
