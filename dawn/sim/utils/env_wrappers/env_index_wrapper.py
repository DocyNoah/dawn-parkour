from __future__ import annotations

from typing import TYPE_CHECKING

from dawn.sim.utils.env_wrappers.helpers import EnvIndexKeyboardController
from dawn.sim.utils.env_wrappers.vec_env_wrapper import VecEnvWrapper

if TYPE_CHECKING:
    import torch

    from dawn.sim.envs.manager_based_rl_amp_env import ManagerBasedRLAMPEnv


class EnvIndexWrapper(VecEnvWrapper):
    def __init__(self, env: ManagerBasedRLAMPEnv | VecEnvWrapper):
        super().__init__(env)
        self._keyboard_controller = EnvIndexKeyboardController(self.num_envs)

    @property
    def env_idx(self) -> int:
        return self._keyboard_controller.env_idx

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs_dict, rew, dones, extras = self.env.step(actions)

        if not self.unwrapped.cfg.sim.headless:
            self._keyboard_controller.setup()

        return obs_dict, rew, dones, extras

    def close(self) -> None:
        self._keyboard_controller.close()
        return self.env.close()
