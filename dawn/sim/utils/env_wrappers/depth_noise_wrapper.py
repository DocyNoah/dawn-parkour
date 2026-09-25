from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from dawn.sim.utils.depth_noise_model import DepthNoise
from dawn.sim.utils.env_wrappers.vec_env_wrapper import VecEnvWrapper
from dawn.sim.utils.image_processing import normalize_depth_image

if TYPE_CHECKING:
    from dawn.sim.envs.manager_based_rl_amp_env import ManagerBasedRLAMPEnv


class DepthNoiseWrapper(VecEnvWrapper):
    def __init__(self, env: ManagerBasedRLAMPEnv | VecEnvWrapper) -> None:
        if getattr(env, "APPLIES_OBSERVATION_SHAPING", False):
            raise ValueError("DepthNoiseWrapper must run before ObservationWrapper.")
        if not hasattr(env, "depth_buffer"):
            raise ValueError("DepthNoiseWrapper requires DepthBufferWrapper to expose depth_buffer.")

        super().__init__(env)
        H, W = self.unwrapped.cfg.depth.resized_image_shape

        self._cached_depth_clean = torch.zeros(self.num_envs, 1, H, W, device=self.device)
        self._cached_depth_noisy = torch.zeros(self.num_envs, 1, H, W, device=self.device)
        self._depth_noise = DepthNoise(near_clip=0.28, far_clip=2.0)

    @property
    def depth_noise(self) -> DepthNoise:
        return self._depth_noise

    def reset(self) -> tuple[torch.Tensor, dict]:
        self._cached_depth_clean.zero_()
        self._cached_depth_noisy.zero_()
        obs_dict, extras = self.env.reset()
        self._attach_depth_pair(obs_dict)
        return obs_dict, extras

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs_dict, rew, dones, extras = self.env.step(actions)

        delayed_depth = extras.pop("delayed_depth", None)

        if delayed_depth is not None:
            self._cache_clean_and_noisy_depth(delayed_depth)

        self._attach_depth_pair(obs_dict)

        return obs_dict, rew, dones, extras

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        obs_dict, extras = self.env.get_observations()
        self._attach_depth_pair(obs_dict)
        return obs_dict, extras

    def _cache_clean_and_noisy_depth(self, delayed_depth: torch.Tensor) -> None:
        clean = delayed_depth
        noisy = self._depth_noise.apply(clean)

        self._cached_depth_clean[self.depth_index] = normalize_depth_image(clean, 0.28, 2.0)
        self._cached_depth_noisy[self.depth_index] = normalize_depth_image(noisy, 0.28, 2.0)

    def _attach_depth_pair(self, obs_dict: dict[str, torch.Tensor]) -> None:
        obs_dict["depth"] = self._cached_depth_noisy
        obs_dict["depth_clean"] = self._cached_depth_clean
