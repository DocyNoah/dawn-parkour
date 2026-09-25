import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dawn.rl.networks.dreamer.sub_networks import ImgChLayerNorm
from dawn.rl.utils import uniform_weight_init, weight_init


class ConvDecoder(nn.Module):
    def __init__(
        self,
        feat_size: int,
        shape: tuple[int, int, int] = (3, 64, 64),
        depth: int = 32,
        act: Any = nn.ELU,
        norm: bool = True,
        kernel_size: int = 4,
        minres: int = 4,
        outscale: float = 1.0,
        cnn_sigmoid: bool = False,
    ) -> None:
        _input_ch, h, w = shape
        stages = int(np.log2(w) - np.log2(minres))
        self.h_list: list[int] = []
        self.w_list: list[int] = []
        for _ in range(stages):
            h, w = (h + 1) // 2, (w + 1) // 2
            self.h_list.append(h)
            self.w_list.append(w)
        self.h_list = self.h_list[::-1]
        self.w_list = self.w_list[::-1]
        self.h_list.append(shape[1])
        self.w_list.append(shape[2])

        super().__init__()
        act = getattr(torch.nn, act)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = len(self.h_list) - 1

        out_ch = self.h_list[0] * self.w_list[0] * depth * 2 ** (len(self.h_list) - 2)
        self._embed_size = out_ch

        self._linear_layer = nn.Linear(feat_size, out_ch)
        self._linear_layer.apply(uniform_weight_init(outscale))

        in_dim = out_ch // (self.h_list[0] * self.w_list[0])
        out_dim = in_dim // 2
        layers: list[nn.Module] = []
        for i in range(layer_num):
            bias = False
            if i == layer_num - 1:
                out_dim = self._shape[0]
                act = False
                bias = True
                norm = False

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth

            if self.h_list[i] * 2 == self.h_list[i + 1]:
                pad_h, outpad_h = 1, 0
            else:
                pad_h, outpad_h = 2, 1
            if self.w_list[i] * 2 == self.w_list[i + 1]:
                pad_w, outpad_w = 1, 0
            else:
                pad_w, outpad_w = 2, 1

            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            if act:
                layers.append(act())
            in_dim = out_dim
            out_dim //= 2

        for m in layers[:-1]:
            m.apply(weight_init)
        layers[-1].apply(uniform_weight_init(outscale))
        self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k: int, s: int, d: int) -> tuple[int, int]:
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features: torch.Tensor, dtype: Any | None = None) -> torch.Tensor:
        x = self._linear_layer(features)
        x = self._to_conv(x, leading_shape=features.shape[:-1])
        x = self.layers(x)
        return self._from_conv(x, leading_shape=features.shape[:-1])

    def _to_conv(self, x: torch.Tensor, leading_shape: tuple[int, ...]) -> torch.Tensor:
        del leading_shape

        return x.reshape([-1, self._embed_size // (self.h_list[0] * self.w_list[0]), self.h_list[0], self.w_list[0]])

    def _from_conv(self, x: torch.Tensor, leading_shape: tuple[int, ...]) -> torch.Tensor:
        mean = x.reshape(leading_shape + self._shape)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean)
        return mean
