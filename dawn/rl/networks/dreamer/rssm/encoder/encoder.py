import torch
from torch import nn

from dawn.rl.networks.dreamer.rssm._layers import linear_norm_act
from dawn.rl.utils import uniform_weight_init, weight_init


class Encoder(nn.Module):
    def __init__(
        self,
        h_dim: int,
        embed_dim: int,
        hidden: int,
        z_dim: int,
        z_classes: int,
        norm: bool,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.z_classes = z_classes

        self.posterior_trunk = linear_norm_act(h_dim + embed_dim, hidden, norm, activation)
        self.posterior_trunk.apply(weight_init)
        self.posterior_head = nn.Linear(hidden, z_dim * z_classes)
        self.posterior_head.apply(uniform_weight_init(1.0))

    def forward(self, h: torch.Tensor, x_embed: torch.Tensor) -> torch.Tensor:
        hidden = self.posterior_trunk(torch.cat([h, x_embed], -1))
        flat = self.posterior_head(hidden)
        return flat.reshape([*hidden.shape[:-1], self.z_dim, self.z_classes])
