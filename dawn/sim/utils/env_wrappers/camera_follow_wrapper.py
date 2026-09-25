from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_apply, yaw_quat

from dawn.sim.utils.env_wrappers.helpers import CameraFollowKeyboardController
from dawn.sim.utils.env_wrappers.vec_env_wrapper import VecEnvWrapper

if TYPE_CHECKING:
    from dawn.sim.envs.manager_based_rl_amp_env import ManagerBasedRLAMPEnv


@dataclass(frozen=True)
class CameraFollowConfig:
    target_height: float = 0.2

    smoothing_ratio: float = 0.6

    start_azimuth_deg: float = 90.0

    azimuth_step_deg: float = 5.0

    start_elevation_deg: float = 11.0

    elevation_step_deg: float = 5.0

    min_elevation_deg: float = 0.0

    max_elevation_deg: float = 85.0

    start_distance: float = 2.6

    distance_step: float = 0.05

    min_distance: float = 1.0

    max_distance: float = 15.0


@dataclass
class CameraFollowState:
    azimuth_deg: float
    elevation_deg: float
    distance: float

    eye: list[float] | None = None
    target: list[float] | None = None

    last_env_idx: int | None = None


class CameraFollowWrapper(VecEnvWrapper):
    def __init__(
        self,
        env: ManagerBasedRLAMPEnv | VecEnvWrapper,
        config: CameraFollowConfig | None = None,
    ):
        super().__init__(env)
        self._keyboard_controller = CameraFollowKeyboardController()
        self._config = config or CameraFollowConfig()
        self._camera = CameraFollowState(
            azimuth_deg=self._config.start_azimuth_deg,
            elevation_deg=self._config.start_elevation_deg,
            distance=self._config.start_distance,
            last_env_idx=self.env_idx,
        )

    @property
    def env_idx(self) -> int:
        return self.env.env_idx

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs_dict, rew, dones, extras = self.env.step(actions)

        if self.unwrapped.cfg.sim.headless:
            return obs_dict, rew, dones, extras

        self._keyboard_controller.setup()
        self._update_follow_camera()
        return obs_dict, rew, dones, extras

    def _update_follow_camera(self) -> None:
        if not self._keyboard_controller.follow_enabled:
            self._camera.eye = None
            self._camera.target = None
            return

        robot = self.unwrapped.scene["robot"]
        root_pos = robot.data.root_pos_w[self.env_idx]
        root_quat = robot.data.root_quat_w[self.env_idx]
        self._update_camera_controls()
        eye, target = self._get_camera_pose(root_pos, root_quat)
        self._smooth_camera_pose(eye, target)
        self.unwrapped.sim.set_camera_view(eye=self._camera.eye, target=self._camera.target)

    def _update_camera_controls(self) -> None:
        horizontal_direction = self._keyboard_controller.horizontal_orbit_direction
        if horizontal_direction != 0:
            self._camera.azimuth_deg = (
                self._camera.azimuth_deg + horizontal_direction * self._config.azimuth_step_deg
            ) % 360.0

        vertical_direction = self._keyboard_controller.vertical_orbit_direction
        if vertical_direction != 0:
            self._camera.elevation_deg += vertical_direction * self._config.elevation_step_deg
            if self._camera.elevation_deg < self._config.min_elevation_deg:
                self._camera.elevation_deg = self._config.min_elevation_deg
            elif self._camera.elevation_deg > self._config.max_elevation_deg:
                self._camera.elevation_deg = self._config.max_elevation_deg

        zoom_direction = self._keyboard_controller.zoom_direction
        if zoom_direction != 0:
            self._camera.distance += zoom_direction * self._config.distance_step
            if self._camera.distance < self._config.min_distance:
                self._camera.distance = self._config.min_distance
            elif self._camera.distance > self._config.max_distance:
                self._camera.distance = self._config.max_distance

    def _get_camera_pose(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
    ) -> tuple[list[float], list[float]]:
        azimuth = math.radians(self._camera.azimuth_deg)
        elevation = math.radians(self._camera.elevation_deg)
        horizontal = self._camera.distance * math.cos(elevation)
        offset = torch.tensor(
            [
                horizontal * math.cos(azimuth),
                horizontal * math.sin(azimuth),
                self._camera.distance * math.sin(elevation),
            ],
            device=root_pos.device,
            dtype=root_pos.dtype,
        )
        eye = root_pos + quat_apply(yaw_quat(root_quat.unsqueeze(0)), offset.unsqueeze(0)).squeeze(0)
        target = root_pos + torch.tensor(
            [0.0, 0.0, self._config.target_height],
            device=root_pos.device,
            dtype=root_pos.dtype,
        )
        return eye.tolist(), target.tolist()

    def _smooth_camera_pose(self, eye: list[float], target: list[float]) -> None:
        if self._camera.eye is None or self._camera.last_env_idx != self.env_idx:
            self._camera.eye = eye
            self._camera.target = target
            self._camera.last_env_idx = self.env_idx
            return

        smoothing = self._config.smoothing_ratio
        self._camera.eye = [
            (1 - smoothing) * previous + smoothing * current
            for previous, current in zip(self._camera.eye, eye, strict=True)
        ]
        self._camera.target = [
            (1 - smoothing) * previous + smoothing * current
            for previous, current in zip(self._camera.target, target, strict=True)
        ]

    def close(self) -> None:
        self._keyboard_controller.close()
        return self.env.close()
