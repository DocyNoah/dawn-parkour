from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Generator


class ReplayBuffer:
    def __init__(self, obs_dim: int, buffer_size: int, device: str = "cpu"):
        self.states = torch.zeros(buffer_size, obs_dim).to(device)
        self.next_states = torch.zeros(buffer_size, obs_dim).to(device)

        self.buffer_size = buffer_size
        self.device = device

        self.step = 0
        self.num_samples = 0

    def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
        num_states = states.shape[0]
        start_idx = self.step
        end_idx = start_idx + num_states

        if end_idx > self.buffer_size:
            self.states[self.step : self.buffer_size] = states[: self.buffer_size - self.step]
            self.next_states[self.step : self.buffer_size] = next_states[: self.buffer_size - self.step]
            self.states[: end_idx - self.buffer_size] = states[self.buffer_size - self.step :]
            self.next_states[: end_idx - self.buffer_size] = next_states[self.buffer_size - self.step :]
        else:
            self.states[start_idx:end_idx] = states
            self.next_states[start_idx:end_idx] = next_states

        self.step = end_idx % self.buffer_size
        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        for _ in range(num_mini_batch):
            sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            yield (self.states[sample_idxs].to(self.device), self.next_states[sample_idxs].to(self.device))
