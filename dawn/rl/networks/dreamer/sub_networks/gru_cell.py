from collections.abc import Callable

import torch
from torch import nn


class GRUCell(nn.Module):
    def __init__(
        self,
        inp_size: int,
        size: int,
        norm: bool = True,
        act: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
        update_bias: float = -1,
    ) -> None:
        super().__init__()
        self._size = size
        self._act = act
        self._update_bias = update_bias
        self.layers = nn.Sequential()
        self.layers.add_module("GRU_linear", nn.Linear(inp_size + size, 3 * size, bias=False))
        if norm:
            self.layers.add_module("GRU_norm", nn.LayerNorm(3 * size, eps=1e-03))

    def forward(self, inputs: torch.Tensor, state: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        state = state[0]
        parts = self.layers(torch.cat([inputs, state], -1))
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]
