from __future__ import annotations

from typing import TYPE_CHECKING

from dawn.sim.utils.env_wrappers.vec_env_wrapper import VecEnvWrapper

if TYPE_CHECKING:
    import torch

    from dawn.sim.envs.manager_based_rl_amp_env import ManagerBasedRLAMPEnv


class ChannelFirstWrapper(VecEnvWrapper):
    _IMAGE_KEY_PARTS = ("depth", "image", "rgb")
    _CHANNEL_COUNTS = (1, 3, 4)

    def __init__(self, env: ManagerBasedRLAMPEnv | VecEnvWrapper):
        super().__init__(env)

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        obs_dict, extras = self.env.get_observations()
        return self._transform_observations(obs_dict, extras), extras

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs_dict, extras = self.env.reset()
        return self._transform_observations(obs_dict, extras), extras

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs_dict, rew, dones, extras = self.env.step(actions)
        return self._transform_observations(obs_dict, extras), rew, dones, extras

    def _transform_observations(
        self,
        obs_dict: dict[str, torch.Tensor],
        extras: dict,
    ) -> dict[str, torch.Tensor]:
        self._to_channel_first(obs_dict)

        raw_obs_dict = extras.get("observations")
        if raw_obs_dict is not None and raw_obs_dict is not obs_dict:
            self._to_channel_first(raw_obs_dict)

        return obs_dict

    def _to_channel_first(self, obs_dict: dict[str, torch.Tensor]) -> None:
        for key, value in obs_dict.items():
            if self._is_channel_last_image(key, value):
                obs_dict[key] = value.movedim(-1, -3).contiguous()

    def _is_channel_last_image(self, key: str, value: torch.Tensor) -> bool:
        if not any(key_part in key for key_part in self._IMAGE_KEY_PARTS):
            return False
        if value.dim() < 4:
            return False
        if value.shape[-1] not in self._CHANNEL_COUNTS:
            return False
        return value.shape[-3] not in self._CHANNEL_COUNTS
