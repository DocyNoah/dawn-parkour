from typing import TYPE_CHECKING

import torch
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

from dawn.sim.utils.image_processing import (
    clip_depth_image,
)

if TYPE_CHECKING:
    from isaaclab.assets.articulation.articulation import Articulation
    from isaaclab.assets.rigid_object.rigid_object import RigidObject
    from isaaclab.sensors import Camera, RayCasterCamera, TiledCamera


def friction(env: ManagerBasedRLEnv) -> torch.Tensor:
    asset: RigidObject = env.scene["robot"]
    materials = asset.root_physx_view.get_material_properties().clone()

    foot_idx = -1

    return materials[:, foot_idx, :].to(env.device)


def mass(env: ManagerBasedRLEnv) -> torch.Tensor:
    asset: RigidObject = env.scene["robot"]
    masses = asset.root_physx_view.get_masses().clone()

    return masses[:, 0].to(env.device).unsqueeze(-1)


def com_pos_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    coms = asset.root_physx_view.get_coms().clone()
    return coms[:, body_ids, :3].flatten(start_dim=1).to(env.device)


def p_gain(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_stiffness[:, asset_cfg.joint_ids]


def d_gain(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_damping[:, asset_cfg.joint_ids]


def image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "depth",
    normalize: bool = True,
) -> torch.Tensor:
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    if data_type != "depth":
        raise ValueError(f"Unsupported image data_type: {data_type}. Only 'depth' is supported.")

    images = sensor.data.output[data_type]

    if normalize:
        images[images == float("inf")] = 100.0

    images = images.clone()

    images = clip_depth_image(images, 0.28, 2.0)

    if hasattr(env, "_depth_camera_active_mask") and env._depth_camera_active_mask is not None:  # noqa: SLF001
        inactive_mask = ~env._depth_camera_active_mask  # noqa: SLF001
        if inactive_mask.any():
            images[inactive_mask] = 0.0

    return images.to(torch.float32)


def contact_footforce(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history

    forces = net_contact_forces[:, 0, sensor_cfg.body_ids, :]

    return forces.flatten(start_dim=1)


def contact_flag(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.1,
) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_flag = torch.norm(net_contact_forces[:, 0, sensor_cfg.body_ids, :], dim=-1) > threshold
    return contact_flag


reorder_indices = torch.tensor([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11])


def joint_pos_amp(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_pos = joint_pos[..., reorder_indices]
    return joint_pos


def joint_vel_amp(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    joint_vel = joint_vel[..., reorder_indices]
    return joint_vel
