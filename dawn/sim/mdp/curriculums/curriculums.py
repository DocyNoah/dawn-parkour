from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from dawn.sim.mdp.terrains.terrain_importer import TerrainImporterWithEdges


def terrain_levels_vel_stats(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporterWithEdges = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)

    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2

    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)

    info = {
        "all_mean": torch.mean(terrain.terrain_levels.float()),
        "all_min": torch.min(terrain.terrain_levels.float()),
        "all_max": torch.max(terrain.terrain_levels.float()),
        "all_median": torch.median(terrain.terrain_levels.float()),
        "all_std": torch.std(terrain.terrain_levels.float()),
    }

    if hasattr(terrain, "terrain_type_names"):
        all_terrain_type_indices = terrain.terrain_types

        terrain_type_indices_np = all_terrain_type_indices.cpu().numpy()

        if isinstance(terrain.terrain_type_names, np.ndarray):
            all_terrain_names = np.array(terrain.terrain_type_names)[terrain_type_indices_np]
        else:
            all_terrain_names = np.array([terrain.terrain_type_names[i] for i in terrain_type_indices_np])

        unique_terrain_type_names = set(terrain.terrain_type_names)
        for terrain_type_name in unique_terrain_type_names:
            terrain_type_mask = torch.tensor(all_terrain_names == terrain_type_name, device=terrain.device)

            if torch.any(terrain_type_mask):
                terrain_type_levels = terrain.terrain_levels[terrain_type_mask].float()
                info[f"{terrain_type_name}_mean"] = torch.mean(terrain_type_levels)
                info[f"{terrain_type_name}_min"] = torch.min(terrain_type_levels)
                info[f"{terrain_type_name}_max"] = torch.max(terrain_type_levels)
                info[f"{terrain_type_name}_median"] = torch.median(terrain_type_levels)
                info[f"{terrain_type_name}_std"] = torch.std(terrain_type_levels)

    return info


def terrain_levels_vel_only_up(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    def update_env_origins(
        terrain: TerrainImporter,
        env_ids: torch.Tensor,
        move_up: torch.Tensor,
        move_down: torch.Tensor,
    ) -> None:
        if terrain.terrain_origins is None:
            return

        terrain.terrain_levels[env_ids] += 1 * move_up - 1 * move_down

        terrain.terrain_levels[env_ids] = torch.clip(terrain.terrain_levels[env_ids], 0, terrain.max_terrain_level - 1)

        terrain.env_origins[env_ids] = terrain.terrain_origins[
            terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
        ]

    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)

    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2

    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    update_env_origins(terrain, env_ids, move_up, move_down)

    return torch.mean(terrain.terrain_levels.float())
