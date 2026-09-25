# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar

import gymnasium as gym
import numpy as np
import torch
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager
from isaaclab.sim import utils as sim_utils
from isaaclab.ui.widgets import ManagerLiveVisualizer
from isaacsim.core.utils import prims as prim_utils  # type: ignore
from isaacsim.core.version import get_version  # type: ignore

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.envs.common import VecEnvStepReturn

    from dawn.sim.envs.manager_based_rl_amp_env_cfg import ManagerBasedRLAMPEnvCfg


class ManagerBasedRLAMPEnv(ManagerBasedEnv, gym.Env):
    is_vector_env: ClassVar[bool] = True

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }

    cfg: ManagerBasedRLAMPEnvCfg

    def __init__(self, cfg: ManagerBasedRLAMPEnvCfg, render_mode: str | None = None, **kwargs):
        self.common_step_counter = 0

        self._iteration: int = 0

        self.episode_length_buf = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.long)

        super().__init__(cfg=cfg)

        self.render_mode = render_mode

        self._depth_camera_active_mask: torch.Tensor | None = None
        if hasattr(cfg.scene, "depth_camera_num") and cfg.scene.depth_camera_num is not None:
            self._configure_depth_camera_distribution(cfg)

        self.metadata["render_fps"] = 1 / self.step_dt

        print("[INFO]: Completed setting up the environment...")

    def _configure_depth_camera_distribution(self, cfg: ManagerBasedRLAMPEnvCfg) -> None:
        depth_camera_num = cfg.scene.depth_camera_num
        num_envs = cfg.scene.num_envs

        if depth_camera_num is None or depth_camera_num >= num_envs:
            self._depth_camera_active_mask = torch.ones(num_envs, device=cfg.sim.device, dtype=torch.bool)
            return

        terrain = self.scene["terrain"]
        if not hasattr(terrain, "terrain_type_names") or terrain.terrain_type_names is None:
            step = num_envs / depth_camera_num
            selected_indices = [int(i * step) for i in range(depth_camera_num)]
            self._depth_camera_active_mask = torch.zeros(num_envs, device=cfg.sim.device, dtype=torch.bool)
            self._depth_camera_active_mask[selected_indices] = True
            return

        terrain_types = terrain.terrain_types
        terrain_type_names = terrain.terrain_type_names

        unique_terrain_names = np.unique(terrain_type_names)
        terrain_counts = {}
        terrain_env_ids = {}

        for terrain_name in unique_terrain_names:
            col_indices = np.where(terrain_type_names == terrain_name)[0]

            env_mask = torch.isin(terrain_types, torch.tensor(col_indices, device=terrain_types.device))
            env_ids = torch.where(env_mask)[0].cpu().numpy()
            terrain_counts[terrain_name] = len(env_ids)
            terrain_env_ids[terrain_name] = env_ids

        total_envs = sum(terrain_counts.values())
        depth_camera_allocations = {}
        allocated_total = 0

        for terrain_name in unique_terrain_names:
            count = terrain_counts[terrain_name]

            allocation = round(depth_camera_num * count / total_envs)

            depth_camera_allocations[terrain_name] = min(allocation, count)
            allocated_total += depth_camera_allocations[terrain_name]

        if allocated_total != depth_camera_num:
            diff = depth_camera_num - allocated_total

            sorted_terrains = sorted(terrain_counts.items(), key=lambda x: x[1], reverse=True)
            for terrain_name, _ in sorted_terrains:
                if diff == 0:
                    break
                current_allocation = depth_camera_allocations[terrain_name]
                max_allocation = terrain_counts[terrain_name]
                if diff > 0 and current_allocation < max_allocation:
                    add = min(diff, max_allocation - current_allocation)
                    depth_camera_allocations[terrain_name] += add
                    diff -= add
                elif diff < 0 and current_allocation > 0:
                    subtract = min(-diff, current_allocation)
                    depth_camera_allocations[terrain_name] -= subtract
                    diff += subtract

        self._depth_camera_active_mask = torch.zeros(num_envs, device=cfg.sim.device, dtype=torch.bool)

        for terrain_name in unique_terrain_names:
            env_ids = terrain_env_ids[terrain_name]
            num_to_select = depth_camera_allocations[terrain_name]

            if num_to_select > 0:
                if num_to_select >= len(env_ids):
                    selected_indices = env_ids
                else:
                    step = len(env_ids) / num_to_select
                    selected_indices = [env_ids[int(i * step)] for i in range(num_to_select)]

                self._depth_camera_active_mask[selected_indices] = True

        print(
            f"[INFO]: Configured {self._depth_camera_active_mask.sum().item()} depth cameras "
            f"out of {num_envs} environments, distributed across terrain types."
        )

        self._remove_inactive_cameras(cfg)

    def _remove_inactive_cameras(self, cfg: ManagerBasedRLAMPEnvCfg) -> None:
        if self._depth_camera_active_mask is None:
            return

        if "depth_camera" not in self.scene.sensors:
            return

        depth_camera_cfg = cfg.scene.depth_camera
        if depth_camera_cfg is None:
            return

        prim_path_pattern = depth_camera_cfg.prim_path

        env_regex_ns = self.scene.env_regex_ns
        prim_path_pattern = prim_path_pattern.replace("{ENV_REGEX_NS}", env_regex_ns)

        all_camera_paths = sim_utils.find_matching_prim_paths(prim_path_pattern)

        inactive_env_ids = torch.where(~self._depth_camera_active_mask)[0].cpu().numpy()

        deleted_count = 0
        for env_id in inactive_env_ids:
            env_prim_path = self.scene.env_prim_paths[env_id]

            camera_path = f"{env_prim_path}/Robot/trunk/custom_cam"

            if camera_path in all_camera_paths:
                try:
                    prim_utils.delete_prim(camera_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"[WARNING]: Failed to delete camera at {camera_path}: {e}")

        if deleted_count > 0:
            print(f"[INFO]: Deleted {deleted_count} depth cameras from inactive environments.")

    @property
    def max_episode_length_s(self) -> float:
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        return math.ceil(self.max_episode_length_s / self.step_dt)

    @property
    def iteration(self) -> int:
        if self._iteration < 0:
            raise RuntimeError(
                "env.iteration was read before any runner assigned it. "
                "Runners must set `self.env.iteration = it` at the start of each "
                "learning-loop iteration."
            )
        return self._iteration

    @iteration.setter
    def iteration(self, value: int) -> None:
        self._iteration = value

    def load_managers(self) -> None:
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        super().load_managers()

        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)

        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)

        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        self._configure_gym_env_spaces()

        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def setup_manager_visualizers(self) -> None:
        self.manager_visualizers = {
            "action_manager": ManagerLiveVisualizer(manager=self.action_manager),
            "observation_manager": ManagerLiveVisualizer(manager=self.observation_manager),
            "command_manager": ManagerLiveVisualizer(manager=self.command_manager),
            "termination_manager": ManagerLiveVisualizer(manager=self.termination_manager),
            "reward_manager": ManagerLiveVisualizer(manager=self.reward_manager),
            "curriculum_manager": ManagerLiveVisualizer(manager=self.curriculum_manager),
        }

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1

            self.action_manager.apply_action()

            self.scene.write_data_to_sim()

            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()

            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()

            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            pre_reset_obs = self.observation_manager.compute(update_history=False)
            self.extras["pre_reset_obs"] = pre_reset_obs

            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)

            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)

        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self.observation_manager.compute(update_history=True)

        self._add_terrain_type_info_to_extras()

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _add_terrain_type_info_to_extras(self) -> None:
        terrain = self.scene.terrain
        if not hasattr(terrain, "terrain_type_names") or terrain.terrain_type_names is None:
            return
        terrain_types = getattr(terrain, "terrain_types", None)
        if terrain_types is None:
            return

        terrain_type_indices = terrain_types.detach().cpu().numpy()
        terrain_type_names = np.asarray(terrain.terrain_type_names)
        self.extras["terrain_type_names"] = terrain_type_names[terrain_type_indices].tolist()

    def render(self, recompute: bool = False) -> np.ndarray | None:
        if not self.sim.has_rtx_sensors() and not recompute:
            self.sim.render()

        if self.render_mode == "human" or self.render_mode is None:
            return None
        elif self.render_mode == "rgb_array":
            if self.sim.render_mode.value < self.sim.RenderMode.PARTIAL_RENDERING.value:
                raise RuntimeError(
                    f"Cannot render '{self.render_mode}' when the simulation render mode is"
                    f" '{self.sim.render_mode.name}'. Please set the simulation render mode to:"
                    f"'{self.sim.RenderMode.PARTIAL_RENDERING.name}' or '{self.sim.RenderMode.FULL_RENDERING.name}'."
                    " If running headless, make sure --enable_cameras is set."
                )

            if not hasattr(self, "_rgb_annotator"):
                import omni.replicator.core as rep  # type: ignore

                self._render_product = rep.create.render_product(
                    self.cfg.viewer.cam_prim_path, self.cfg.viewer.resolution
                )

                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])

            rgb_data = self._rgb_annotator.get_data()

            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)

            if rgb_data.size == 0:
                return np.zeros((self.cfg.viewer.resolution[1], self.cfg.viewer.resolution[0], 3), dtype=np.uint8)
            else:
                return rgb_data[:, :, :3]
        else:
            raise NotImplementedError(
                f"Render mode '{self.render_mode}' is not supported. Please use: {self.metadata['render_modes']}."
            )

    def close(self) -> None:
        if not self._is_closed:
            del self.command_manager
            del self.reward_manager
            del self.termination_manager
            del self.curriculum_manager

            super().close()

    def _configure_gym_env_spaces(self) -> None:
        self.single_observation_space = gym.spaces.Dict()
        for group_name, group_term_names in self.observation_manager.active_terms.items():
            has_concatenated_obs = self.observation_manager.group_obs_concatenate[group_name]
            group_dim = self.observation_manager.group_obs_dim[group_name]

            if has_concatenated_obs:
                self.single_observation_space[group_name] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=group_dim)
            else:
                group_term_cfgs = self.observation_manager._group_obs_term_cfgs[group_name]  # noqa: SLF001
                term_dict = {}
                for term_name, term_dim, term_cfg in zip(group_term_names, group_dim, group_term_cfgs):
                    low = -np.inf if term_cfg.clip is None else term_cfg.clip[0]
                    high = np.inf if term_cfg.clip is None else term_cfg.clip[1]
                    term_dict[term_name] = gym.spaces.Box(low=low, high=high, shape=term_dim)
                self.single_observation_space[group_name] = gym.spaces.Dict(term_dict)

        action_dim = sum(self.action_manager.action_term_dim)
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,))

        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        self.curriculum_manager.compute(env_ids=env_ids)

        self.scene.reset(env_ids)

        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        self.extras["log"] = {}

        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)

        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)

        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)

        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)

        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)

        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)

        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)

        info = self.recorder_manager.reset(env_ids)
        self.extras["log"].update(info)

        self.episode_length_buf[env_ids] = 0
