from __future__ import annotations

from typing import NamedTuple

import torch


class Latent(NamedTuple):
    h: torch.Tensor
    z: torch.Tensor

    @property
    def s(self) -> torch.Tensor:
        z_flat = self.z.reshape([*self.z.shape[:-2], self.z.shape[-2] * self.z.shape[-1]])
        return torch.cat([z_flat, self.h], -1)


class LatentTraj(NamedTuple):
    h: torch.Tensor
    z: torch.Tensor

    @property
    def s(self) -> torch.Tensor:
        z_flat = self.z.reshape([*self.z.shape[:-2], self.z.shape[-2] * self.z.shape[-1]])
        return torch.cat([z_flat, self.h], -1)
