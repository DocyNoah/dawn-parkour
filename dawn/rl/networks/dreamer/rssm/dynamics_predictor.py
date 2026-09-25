import torch
from torch import nn

from dawn.rl.networks.dreamer.rssm._layers import linear_norm_act
from dawn.rl.utils import uniform_weight_init, weight_init


class DynamicsPredictor(nn.Module):
    def __init__(
        self,
        h_dim: int,
        hidden: int,
        z_dim: int,
        z_classes: int,
        norm: bool,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.z_classes = z_classes

        self.trunk = linear_norm_act(h_dim, hidden, norm, activation)
        self.trunk.apply(weight_init)
        self.head = nn.Linear(hidden, z_dim * z_classes)
        self.head.apply(uniform_weight_init(1.0))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        flat = self.head(self.trunk(h))
        return flat.reshape([*flat.shape[:-1], self.z_dim, self.z_classes])
