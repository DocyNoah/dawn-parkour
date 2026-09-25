from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from dawn.sim.utils.env_wrappers.vec_env_wrapper import VecEnvWrapper

if TYPE_CHECKING:
    from dawn.sim.envs.manager_based_rl_amp_env import ManagerBasedRLAMPEnv


class DepthBufferWrapper(VecEnvWrapper):
    def __init__(self, env: ManagerBasedRLAMPEnv | VecEnvWrapper):
        current = env
        while isinstance(current, VecEnvWrapper):
            if getattr(current, "APPLIES_OBSERVATION_SHAPING", False):
                raise ValueError(
                    "DepthBufferWrapper must run before ObservationWrapper so delayed depth uses raw observations."
                )
            current = current.env

        super().__init__(env)

        depth_cfg = getattr(self.unwrapped.cfg, "depth", None)
        if depth_cfg is None:
            raise ValueError("DepthBufferWrapper requires env.cfg.depth.")

        self.global_step = 0
        camera_num_envs = self.num_envs
        if "depth_camera" in self.unwrapped.scene.sensors:
            sensor = self.unwrapped.scene.sensors["depth_camera"]
            if hasattr(sensor, "_view") and hasattr(sensor._view, "count"):  # noqa: SLF001
                camera_num_envs = sensor._view.count  # noqa: SLF001
        else:
            cfg_camera_num_envs = getattr(depth_cfg, "camera_num_envs", None)
            if cfg_camera_num_envs is not None:
                camera_num_envs = int(cfg_camera_num_envs)

        H, W = depth_cfg.resized_image_shape

        self.depth_buffer = torch.zeros(
            camera_num_envs,
            depth_cfg.buffer_len,
            1,
            H,
            W,
            device=self.device,
        )

        self._depth_index = np.arange(camera_num_envs)

        self._cached_depth = torch.zeros(self.num_envs, 1, H, W, device=self.device)

    @property
    def depth_index(self) -> np.ndarray:
        return self._depth_index

    def reset(self) -> tuple[torch.Tensor, dict]:
        self.global_step = 0
        self.depth_buffer.zero_()
        self._cached_depth.zero_()
        obs_dict, extras = self.env.reset()
        self._attach_depth(obs_dict)
        return obs_dict, extras

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        obs_dict, extras = self.env.get_observations()
        self._attach_depth(obs_dict)
        return obs_dict, extras

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self.global_step += 1
        obs_dict, rew, dones, extras = self.env.step(actions)

        raw_obs_dict = extras.get("observations", obs_dict)
        self._update_depth_buffer(raw_obs_dict)

        if self._should_update_depth_buffer():
            delayed_depth = self.depth_buffer[:, -2]
            extras["delayed_depth"] = delayed_depth
            self._cache_delayed_depth(delayed_depth)
        else:
            extras["delayed_depth"] = None

        self._attach_depth(obs_dict)

        return obs_dict, rew, dones, extras

    def _cache_delayed_depth(self, delayed_depth: torch.Tensor) -> None:
        self._cached_depth[self._depth_index] = delayed_depth

    def _attach_depth(self, obs_dict: dict[str, torch.Tensor]) -> None:
        obs_dict["depth"] = self._cached_depth

    def _update_depth_buffer(self, obs_dict: dict[str, torch.Tensor]) -> None:
        if not self._should_update_depth_buffer():
            return

        depth_image = obs_dict["depth"]
        init_flag = self.unwrapped.episode_length_buf <= 1

        for buffer_idx in range(len(self._depth_index)):
            self._update_single_env_depth_buffer(
                buffer_idx=buffer_idx,
                depth_frame=depth_image[buffer_idx],
                is_episode_start=bool(init_flag[buffer_idx]),
            )

    def _should_update_depth_buffer(self) -> bool:
        return self.global_step % self.unwrapped.cfg.depth.update_interval == 0

    def _update_single_env_depth_buffer(
        self,
        buffer_idx: int,
        depth_frame: torch.Tensor,
        is_episode_start: bool,
    ) -> None:
        if is_episode_start:
            self.depth_buffer[buffer_idx] = depth_frame.unsqueeze(0).expand(
                self.unwrapped.cfg.depth.buffer_len, -1, -1, -1
            )
            return

        self.depth_buffer[buffer_idx] = torch.cat(
            [self.depth_buffer[buffer_idx, 1:], depth_frame.unsqueeze(0).to(self.device)],
            dim=0,
        )
