import gymnasium as gym
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

from dawn.sim.envs.manager_based_rl_amp_env import ManagerBasedRLAMPEnv
from dawn.sim.utils.env_wrappers.vec_env_wrapper import VecEnvWrapper


class IsaaclabWrapper(VecEnvWrapper):
    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv | ManagerBasedRLAMPEnv):
        if (
            not isinstance(env.unwrapped, ManagerBasedRLEnv)
            and not isinstance(env.unwrapped, DirectRLEnv)
            and not isinstance(env.unwrapped, ManagerBasedRLAMPEnv)
        ):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )

        super().__init__(env)

    @property
    def device(self) -> torch.device:
        return self.unwrapped.device

    @property
    def cfg(self) -> dict | object:
        return self.unwrapped.cfg

    @property
    def num_envs(self) -> int:
        return self.unwrapped.num_envs

    @property
    def num_actions(self) -> int:
        if hasattr(self.unwrapped, "action_manager"):
            return self.unwrapped.action_manager.total_action_dim
        return gym.spaces.flatdim(self.unwrapped.single_action_space)

    @property
    def max_episode_length(self) -> int | torch.Tensor:
        return self.unwrapped.max_episode_length

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.unwrapped.episode_length_buf = value

    @property
    def render_mode(self) -> str | None:
        return self.unwrapped.render_mode

    @property
    def dof_pos_limits(self) -> torch.Tensor:
        if hasattr(self.unwrapped, "scene"):
            if hasattr(self.unwrapped.scene, "articulations") and "robot" in self.unwrapped.scene.articulations:
                robot = self.unwrapped.scene.articulations["robot"]
                return robot.data.soft_joint_pos_limits
            elif hasattr(self.unwrapped.scene, "__getitem__"):
                try:
                    robot = self.unwrapped.scene["robot"]
                    return robot.data.soft_joint_pos_limits
                except (KeyError, TypeError):
                    pass

        if hasattr(self.unwrapped, "_robot"):
            return self.unwrapped._robot.data.soft_joint_pos_limits  # noqa: SLF001
        raise AttributeError(f"Cannot access dof_pos_limits from environment type: {type(self.unwrapped)}")

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()  # noqa
        return obs_dict, {}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        extras["observations"] = obs_dict
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        env_ids = self.unwrapped.reset_buf.nonzero(as_tuple=False).flatten()
        extras["reset_env_ids"] = env_ids
        return obs_dict, rew, dones, extras

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def close(self) -> None:
        return self.env.close()
