import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dawn.rl.utils import uniform_weight_init, weight_init


class DepthPredictor(nn.Module):
    def __init__(
        self,
        forward_heightmap_dim: int = 525,
        prop_dim: int = 33,
        depth_image_dims: tuple[int, int] = (64, 64),
        encoder_hidden_dims: tuple[int, ...] = (256, 128),
        cnn_base_channels: int = 32,
        activation_fn: nn.Module = nn.ELU(),
        norm: bool = True,
        kernel_size: int = 4,
        minres: int = 4,
        outscale: float = 1.0,
        cnn_sigmoid: bool = False,
    ) -> None:
        h, w = depth_image_dims
        stages = int(np.log2(w) - np.log2(minres))
        self.h_list = []
        self.w_list = []

        for i in range(stages):
            h, w = (h + 1) // 2, (w + 1) // 2
            self.h_list.append(h)
            self.w_list.append(w)

        self.h_list = self.h_list[::-1]
        self.w_list = self.w_list[::-1]
        self.h_list.append(depth_image_dims[0])
        self.w_list.append(depth_image_dims[1])

        super().__init__()

        self._cnn_sigmoid = cnn_sigmoid

        layer_num = len(self.h_list) - 1

        out_ch = self.h_list[0] * self.w_list[0] * cnn_base_channels * 2 ** (len(self.h_list) - 2)
        self._embed_size = out_ch

        in_dim = out_ch // (self.h_list[0] * self.w_list[0])
        out_dim = in_dim // 2
        act = getattr(torch.nn, "ELU")

        encoder_layers = []
        encoder_layers.append(nn.Linear(forward_heightmap_dim + prop_dim, encoder_hidden_dims[0]))
        encoder_layers.append(activation_fn)
        for l in range(len(encoder_hidden_dims)):  # noqa: E741
            if l == len(encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], self._embed_size))
            else:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
                encoder_layers.append(activation_fn)
        self.encoder = nn.Sequential(*encoder_layers)

        layers = []

        for i in range(layer_num):
            bias = False
            if i == layer_num - 1:
                out_dim = 1
                act = False
                bias = True
                norm = False

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * cnn_base_channels
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
                layers.append(activation_fn)
            in_dim = out_dim
            out_dim //= 2
        [m.apply(weight_init) for m in layers[:-1]]
        layers[-1].apply(uniform_weight_init(outscale))
        self.decoder = nn.Sequential(*layers)

    def forward(self, forward_heightmap: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        x = torch.concat([forward_heightmap, prop], dim=-1)
        x = self.encoder(x)

        x = x.reshape([-1, self._embed_size // (self.h_list[0] * self.w_list[0]), self.h_list[0], self.w_list[0]])

        x = self.decoder(x)
        mean = x
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean)
        return mean

    def save(self, model_dir: str) -> None:
        depth_predictor_path = os.path.join(model_dir, "depth_predictor.pt")
        torch.save(self.state_dict(), depth_predictor_path)

    def load(self, model_dir: str) -> None:
        depth_predictor_path = os.path.join(model_dir, "depth_predictor.pt")
        self.load_state_dict(torch.load(depth_predictor_path, map_location="cpu", weights_only=True), strict=True)


class ImgChLayerNorm(nn.Module):
    def __init__(self, ch: int, eps: float = 1e-03) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
