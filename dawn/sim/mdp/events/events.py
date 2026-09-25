from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz
from isaacsim.core.prims import XFormPrim  # type: ignore

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class randomize_rigid_body_material(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if not isinstance(self.asset, (RigidObject, Articulation)):
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_material' not supported for asset: '{self.asset_cfg.name}'"
                f" with type: '{type(self.asset)}'."
            )

        if isinstance(self.asset, Articulation) and self.asset_cfg.body_ids != slice(None):
            self.num_shapes_per_body = []
            for link_path in self.asset.root_physx_view.link_paths[0]:
                link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # noqa
                self.num_shapes_per_body.append(link_physx_view.max_shapes)

            num_shapes = sum(self.num_shapes_per_body)
            expected_shapes = self.asset.root_physx_view.max_shapes
            if num_shapes != expected_shapes:
                raise ValueError(
                    "Randomization term 'randomize_rigid_body_material' failed to parse the number of shapes per body."
                    f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
                )
        else:
            self.num_shapes_per_body = None

        friction_range = cfg.params.get("friction_range", (1.0, 1.0))
        friction_scale_factor = cfg.params.get("friction_scale_factor", (1.1, 1.5))

        restitution_range = cfg.params.get("restitution_range", (0.0, 0.0))
        num_buckets = int(cfg.params.get("num_buckets", 1))

        range_list = [friction_range, friction_scale_factor, restitution_range]
        ranges = torch.tensor(range_list, device="cpu")
        buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_buckets, 3), device="cpu")
        friction_buckets, scale_buckets, restitution_buckets = buckets[:, 0], buckets[:, 1], buckets[:, 2]
        self.material_buckets = torch.stack(
            [
                (friction_buckets * scale_buckets),
                friction_buckets,
                restitution_buckets,
            ],
            dim=-1,
        )

        make_consistent = cfg.params.get("make_consistent", False)
        if make_consistent:
            self.material_buckets[:, 1] = torch.min(self.material_buckets[:, 0], self.material_buckets[:, 1])

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        friction_range: tuple[float, float],
        friction_scale_factor: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        total_num_shapes = self.asset.root_physx_view.max_shapes
        bucket_ids = torch.randint(0, num_buckets, (len(env_ids), total_num_shapes), device="cpu")
        material_samples = self.material_buckets[bucket_ids]

        materials = self.asset.root_physx_view.get_material_properties()

        if self.num_shapes_per_body is not None:
            already_sampled = False
            for body_id in self.asset_cfg.body_ids:
                start_idx = sum(self.num_shapes_per_body[:body_id])
                end_idx = start_idx + self.num_shapes_per_body[body_id]

                if not already_sampled:
                    sampled_materials = material_samples[:, start_idx:end_idx]
                    already_sampled = True
                materials[env_ids, start_idx:end_idx] = sampled_materials
        else:
            materials[env_ids] = material_samples[:]

        self.asset.root_physx_view.set_material_properties(materials, env_ids)


default_com = None


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
) -> None:
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), len(body_ids), 3), device="cpu")

    global default_com
    if default_com is None:
        default_com = asset.root_physx_view.get_coms().clone()
    coms = default_com.clone()

    coms[env_ids[:, None], body_ids, :3] += rand_samples

    asset.root_physx_view.set_coms(coms, env_ids)


def randomize_obs_delay_steps(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    obs_delay_range_steps: tuple[int, int],
) -> None:
    def _randint(low: int, high: int, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        return torch.randint(low, high + 1, shape, device=device)

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    else:
        env_ids = env_ids.cpu()

    if "obs_delay_steps" not in env.extras:
        env.extras["obs_delay_steps"] = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env.extras["max_obs_delay_step"] = obs_delay_range_steps[1]

    env.extras["obs_delay_steps"][env_ids] = _randint(*obs_delay_range_steps, (len(env_ids),), device=env.device)


def randomize_camera_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    position_range: dict[str, tuple[float, float]],
    orientation_range: dict[str, tuple[float, float]],
    sensor_cfg: SceneEntityCfg,
) -> None:
    if sensor_cfg.name not in env.scene.sensors:
        return

    camera = env.scene[sensor_cfg.name]

    max_camera_env_id = camera._view.count - 1  # noqa: SLF001
    if env_ids is None:
        env_ids = torch.arange(camera._view.count, device=camera.device)  # noqa: SLF001
    else:
        env_ids = env_ids.to(camera.device)

        env_ids = env_ids[env_ids <= max_camera_env_id]
        if len(env_ids) == 0:
            return

    range_list = [position_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=camera.device)
    rand_pos_offsets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=camera.device)

    range_list = [orientation_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=camera.device)
    rand_euler_offsets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=camera.device)

    rand_quat_offsets = quat_from_euler_xyz(
        rand_euler_offsets[:, 0], rand_euler_offsets[:, 1], rand_euler_offsets[:, 2]
    )

    parent_prim_path = "/".join(camera.cfg.prim_path.rsplit("/", 1)[:-1])

    parent_view = XFormPrim(parent_prim_path, reset_xform_properties=False)
    parent_view.initialize()
    parent_pos_w, parent_quat_w = parent_view.get_world_poses(env_ids)

    new_pos_w = parent_pos_w + math_utils.quat_apply(parent_quat_w, rand_pos_offsets)

    new_quat_w = math_utils.quat_mul(parent_quat_w, rand_quat_offsets)

    camera.set_world_poses(positions=new_pos_w, orientations=new_quat_w, env_ids=env_ids, convention="world")
