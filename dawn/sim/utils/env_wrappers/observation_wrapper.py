from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from dawn.sim.utils.env_wrappers.vec_env_wrapper import VecEnvWrapper

if TYPE_CHECKING:
    from dawn.sim.envs.manager_based_rl_amp_env import ManagerBasedRLAMPEnv


class ObservationWrapper(VecEnvWrapper):
    APPLIES_OBSERVATION_SHAPING = True

    def __init__(self, env: ManagerBasedRLAMPEnv | VecEnvWrapper):
        super().__init__(env)

        self._delayed_obs_keys = self._get_delayed_obs_keys()
        self._delayed_obs_keys = [i for i in self._delayed_obs_keys if "action" not in i]

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        obs_dict, extras = self.env.get_observations()
        return self._transform_observations(obs_dict), extras

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs_dict, extras = self.env.reset()
        return self._transform_observations(obs_dict), extras

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs_dict, rew, dones, extras = self.env.step(actions)
        return self._transform_observations(obs_dict), rew, dones, extras

    @staticmethod
    def _squeeze_channel(obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        depth_key = None
        if "depth" in obs_dict:
            depth_key = "depth"
        elif "depth_hist" in obs_dict:
            depth_key = "depth_hist"
        if depth_key is None:
            return obs_dict
        depth_obs = obs_dict[depth_key]

        if depth_obs.dim() == 5:
            depth_obs = depth_obs.flatten(start_dim=1, end_dim=2)
            obs_dict[depth_key] = depth_obs
        return obs_dict

    def _transform_observations(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        obs_dict = self._squeeze_channel(obs_dict)
        obs_dict = self._get_delayed_obs(obs_dict)
        for key in obs_dict.keys():
            if obs_dict[key].dim() <= 2:
                continue
            if "depth" in key:
                continue
            obs_dict[key] = obs_dict[key].squeeze(1)
        return obs_dict

    def _get_delayed_obs(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "obs_delay_steps" not in self.unwrapped.extras:
            return obs_dict
        max_obs_delay_step = self.unwrapped.extras["max_obs_delay_step"]
        obs_delay_steps = self.unwrapped.extras["obs_delay_steps"]

        delayed_obs_dict = {}
        for obs_key, obs in obs_dict.items():
            if obs_key in self._delayed_obs_keys:
                delayed_obs_dict[obs_key] = self._select_delayed_history(obs, max_obs_delay_step, obs_delay_steps)
            else:
                delayed_obs_dict[obs_key] = obs

        return delayed_obs_dict

    def _select_delayed_history(
        self,
        obs: torch.Tensor,
        max_obs_delay_step: int,
        obs_delay_steps: torch.Tensor,
    ) -> torch.Tensor:
        num_envs, history_length = obs.shape[:2]
        model_history_length = history_length - max_obs_delay_step
        env_indices = torch.arange(num_envs, device=self.device).unsqueeze(1)
        start_indices = (max_obs_delay_step - obs_delay_steps).unsqueeze(1)
        time_offsets = torch.arange(model_history_length, device=self.device).unsqueeze(0)
        time_indices = start_indices + time_offsets
        return obs[env_indices, time_indices, :]

    def _get_delayed_obs_keys(self) -> list[str]:
        delayed_obs_keys = []
        obs_cfg = self.unwrapped.observation_manager.cfg
        if isinstance(obs_cfg, dict):
            obs_cfg_items = obs_cfg.items()
        else:
            obs_cfg_items = obs_cfg.__dict__.items()
        for group_name, group_cfg in obs_cfg_items:
            if group_cfg is None:
                continue
            if isinstance(group_cfg, dict):
                group_cfg_items = group_cfg.items()
            else:
                group_cfg_items = group_cfg.__dict__.items()
            for term_name, term_cfg in group_cfg_items:
                if term_name in [
                    "enable_corruption",
                    "concatenate_terms",
                    "history_length",
                    "flatten_history_dim",
                    "concatenate_dim",
                ]:
                    continue
                if term_cfg is None:
                    continue

                from isaaclab.managers import ObservationTermCfg

                if not isinstance(term_cfg, ObservationTermCfg):
                    continue
                if term_cfg.history_length > 0:
                    delayed_obs_keys.append(group_name)
                    break
        return delayed_obs_keys
