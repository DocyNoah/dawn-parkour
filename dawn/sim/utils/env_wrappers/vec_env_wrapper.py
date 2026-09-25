from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

    from dawn.sim.envs.manager_based_rl_amp_env import ManagerBasedRLAMPEnv


class VecEnvWrapper:
    def __init__(self, env: ManagerBasedRLAMPEnv | VecEnvWrapper):
        self.env = env

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        return str(self)

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv | ManagerBasedRLAMPEnv:
        return self.env.unwrapped

    @property
    def device(self) -> torch.device:
        return self.env.device

    @property
    def cfg(self) -> dict | object:
        return self.env.cfg

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def num_actions(self) -> int:
        return self.env.num_actions

    @property
    def max_episode_length(self) -> int | torch.Tensor:
        return self.env.max_episode_length

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.episode_length_buf = value

    @property
    def iteration(self) -> int:
        return self.env.iteration

    @iteration.setter
    def iteration(self, value: int) -> None:
        self.env.iteration = value

    @property
    def render_mode(self) -> str | None:
        return self.env.render_mode

    @property
    def dof_pos_limits(self) -> torch.Tensor:
        return self.env.dof_pos_limits

    @property
    def depth_index(self) -> torch.Tensor:
        return self.env.depth_index

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        return self.env.get_observations()

    def reset(self) -> tuple[torch.Tensor, dict]:
        return self.env.reset()

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        return self.env.step(actions)

    def seed(self, seed: int = -1) -> int:
        return self.env.seed(seed)

    def close(self) -> None:
        return self.env.close()
