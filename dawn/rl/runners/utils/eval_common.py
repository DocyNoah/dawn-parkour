from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from dawn.sim.utils.env_wrappers import VecEnvWrapper


class SimulationApp(Protocol):
    def is_running(self) -> bool: ...


@dataclass
class RewardTracker:
    env: VecEnvWrapper
    episode_count: int = 0
    step_count: int = 0

    def __post_init__(self) -> None:
        self.rewards = deque(maxlen=100)
        self.current_reward_sum = torch.zeros(
            self.env.num_envs,
            dtype=torch.float,
            device=self.env.device,
        )

    def update(self, reward: torch.Tensor, dones: torch.Tensor) -> None:
        self.step_count += 1
        self.episode_count += (dones > 0).sum().item()
        self.current_reward_sum += reward

        reset_ids = (dones > 0).nonzero(as_tuple=False)

        self.rewards.extend(self.current_reward_sum[reset_ids][:, 0].cpu().numpy().tolist())
        self.current_reward_sum[reset_ids] = 0.0

    def maybe_print(self) -> None:
        if self.step_count % 100 != 0 or len(self.rewards) == 0:
            return

        print(
            f"Total Episodes: {self.episode_count:5d} | "
            f"{len(self.rewards):3d} Episode Mean Reward: {statistics.mean(self.rewards):5.2f}"
        )
