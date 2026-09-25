from typing import Any

import numpy as np
import torch
from torch import distributions as torchd
from torch import nn

from dawn.rl.networks.dreamer.distributions import Bernoulli, DiscDist, SymlogDist, symlog
from dawn.rl.utils import uniform_weight_init, weight_init


class MLP(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        shape: int | tuple[int, ...] | dict[str, tuple[int, ...]] | None,
        layers: int,
        units: int,
        act: str = "SiLU",
        norm: bool = True,
        dist: str = "symlog_mse",
        outscale: float = 1.0,
        symlog_inputs: bool = False,
        device: str | torch.device = "cuda",
        name: str = "NoName",
    ) -> None:
        super().__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        act = getattr(torch.nn, act)
        self._dist = dist
        self._symlog_inputs = symlog_inputs
        self._device = device

        self.layers = nn.Sequential()
        for i in range(layers):
            self.layers.add_module(f"{name}_linear{i}", nn.Linear(inp_dim, units, bias=False))
            if norm:
                self.layers.add_module(f"{name}_norm{i}", nn.LayerNorm(units, eps=1e-03))
            self.layers.add_module(f"{name}_act{i}", act())
            if i == 0:
                inp_dim = units
        self.layers.apply(weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(uniform_weight_init(outscale))

    def forward(self, features: torch.Tensor, dtype: Any | None = None) -> Any:
        x = features

        if self._symlog_inputs:
            x = symlog(x)
        out = self.layers(x)

        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                dists.update({name: self.dist(self._dist, mean, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            return self.dist(self._dist, mean, self._shape)

    def dist(self, dist: str, mean: torch.Tensor, shape: tuple[int, ...]) -> Any:
        if dist == "binary":
            return Bernoulli(torchd.independent.Independent(torchd.bernoulli.Bernoulli(logits=mean), len(shape)))
        if dist == "symlog_disc":
            return DiscDist(logits=mean, device=self._device)
        if dist == "symlog_mse":
            return SymlogDist(mean)
        raise NotImplementedError(dist)
