import torch
from torch import nn

from dawn.rl.networks.dreamer.rssm._layers import linear_norm_act
from dawn.rl.networks.dreamer.sub_networks import GRUCell
from dawn.rl.utils import weight_init


class SequenceModel(nn.Module):
    def __init__(
        self,
        z_flat_dim: int,
        action_dim: int,
        hidden: int,
        h_dim: int,
        norm: bool,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        self.pre_gru = linear_norm_act(z_flat_dim + action_dim, hidden, norm, activation)
        self.pre_gru.apply(weight_init)
        self.gru = GRUCell(hidden, h_dim, norm=norm)
        self.gru.apply(weight_init)

    def forward(
        self,
        prev_z_flat: torch.Tensor,
        prev_action: torch.Tensor,
        prev_h: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([prev_z_flat, prev_action], -1)
        x = self.pre_gru(x)

        _, h = self.gru(x, [prev_h])
        return h[0]
