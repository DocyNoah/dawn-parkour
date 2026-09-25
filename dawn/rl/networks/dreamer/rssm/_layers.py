from torch import nn


def linear_norm_act(in_dim: int, out_dim: int, norm: bool, activation: nn.Module) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, out_dim, bias=False)]
    if norm:
        layers.append(nn.LayerNorm(out_dim, eps=1e-03))
    layers.append(activation)
    return nn.Sequential(*layers)
