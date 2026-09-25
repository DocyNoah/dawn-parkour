from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from dawn.rl.modules import ActorCritic
from dawn.rl.runners.utils import RewardTracker, SimulationApp

if TYPE_CHECKING:
    from collections.abc import Callable

    from dawn.rl.configs.train_cfg import DawnTrainCfg
    from dawn.sim.utils.env_wrappers import VecEnvWrapper


class EvalRunner:
    def __init__(
        self,
        env: VecEnvWrapper,
        train_cfg: DawnTrainCfg,
        device: str = "cpu",
    ):
        self.cfg = train_cfg
        self.policy_cfg = train_cfg.policy
        self.runner_cfg = train_cfg.runner
        self.device = device
        self.env = env

        self.env.iteration = 0

        obs_dict, _ = self.env.get_observations()
        obs_dict = self._prepare_actor_obs(obs_dict)

        self.actor = ActorCritic(
            obs_dict=obs_dict,
            num_actions=self.env.num_actions,
            use_critic=False,
            policy_cfg=self.policy_cfg,
        ).to(self.device)
        print(self.actor)

        self.num_steps_per_env = self.runner_cfg.num_steps_per_env

        self.eval_mode()

    def _prepare_actor_obs(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return obs_dict

    def load(self, model_dir: str) -> None:
        print("[INFO]: ========== Start loading actor model ==========")
        self.actor.load(model_dir)
        print("[INFO]: ========== Done loading actor model ==========")

    def get_inference_policy(self, device: str | None = None) -> Callable[[torch.Tensor], torch.Tensor]:
        if device is not None:
            self.actor.to(device)
        return self.actor.act_inference

    def eval_mode(self) -> None:
        self.actor.eval_mode()

    def run(self, simulation_app: SimulationApp, real_time: bool, video_length: int | None) -> None:
        dt = self.env.unwrapped.physics_dt
        timestep = 0
        tracker = RewardTracker(env=self.env)
        obs_dict, _ = self.env.get_observations()

        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                actions = self.actor.act_inference(obs_dict)
                obs_dict, reward, dones, _ = self.env.step(actions)
                self.actor.reset(dones)

            tracker.update(reward, dones)
            tracker.maybe_print()

            timestep += 1
            if video_length is not None and timestep == video_length:
                break

            sleep_time = dt - (time.time() - start_time)
            if real_time and sleep_time > 0:
                time.sleep(sleep_time)
