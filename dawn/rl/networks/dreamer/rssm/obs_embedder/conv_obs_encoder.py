import numpy as np
import torch
from torch import nn

from dawn.rl.networks.dreamer.sub_networks import Conv2dSamePad, ImgChLayerNorm
from dawn.rl.utils import weight_init


class ConvObsEncoder(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        depth: int = 32,
        act: str = "SiLU",
        norm: bool = True,
        kernel_size: int = 4,
        minres: int = 4,
    ) -> None:
        super().__init__()
        act = getattr(torch.nn, act)
        input_ch, h, w = input_shape

        stages = int(np.log2(w) - np.log2(minres))

        in_dim = input_ch
        out_dim = depth
        layers: list[nn.Module] = []
        for _ in range(stages):
            layers.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            layers.append(act())
            in_dim = out_dim
            out_dim *= 2
            h, w = (h + 1) // 2, (w + 1) // 2

        self.outdim: int = out_dim // 2 * h * w
        self.layers = nn.Sequential(*layers)
        self.layers.apply(weight_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x, leading_shape = self._to_conv(obs)
        x = self.layers(x)
        return self._from_conv(x, leading_shape)

    @staticmethod
    def _to_conv(obs: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        obs -= 0.5
        leading_shape = tuple(obs.shape[:-3])
        x = obs.reshape((-1, *tuple(obs.shape[-3:])))
        return x, leading_shape

    @staticmethod
    def _from_conv(x: torch.Tensor, leading_shape: tuple[int, ...]) -> torch.Tensor:
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        return x.reshape([*leading_shape, x.shape[-1]])
