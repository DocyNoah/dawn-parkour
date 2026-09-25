from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    def __init__(
        self,
        shape: int | tuple[int, ...],
        eps: float = 1e-2,
        until: int | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

        if self.training:
            self.forward = self._forward_training
        else:
            self.forward = self._forward_eval

    @property
    def mean(self) -> torch.Tensor:
        return self._mean.squeeze(0).clone()

    @property
    def std(self) -> torch.Tensor:
        return self._std.squeeze(0).clone()

    def _forward_training(self, x: torch.Tensor) -> torch.Tensor:
        self.update(x)
        return (x - self._mean) / (self._std + self.eps)

    def _forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Forward implementation required")

    def train(self, mode: bool = True) -> EmpiricalNormalization:
        super().train(mode)
        if mode:
            self.forward = self._forward_training
        else:
            self.forward = self._forward_eval
        return self

    def eval(self) -> EmpiricalNormalization:
        return self.train(False)

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count

        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)
