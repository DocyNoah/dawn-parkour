import re
from typing import Any

import torch
from torch import distributions as torchd
from torch import nn

from dawn.rl.networks.dreamer.distributions import ContDist, MSEDist
from dawn.rl.networks.dreamer.mlp import MLP
from dawn.rl.networks.dreamer.rssm.decoder.conv_decoder import ConvDecoder


class Decoder(nn.Module):
    def __init__(
        self,
        feat_size: int,
        shapes: dict[str, tuple[int, ...]],
        mlp_keys: str,
        cnn_keys: str,
        act: str,
        norm: bool,
        cnn_depth: int,
        kernel_size: int,
        minres: int,
        mlp_layers: int,
        mlp_units: int,
        cnn_sigmoid: bool,
        image_dist: str,
        vector_dist: str,
        outscale: float,
    ) -> None:
        super().__init__()

        excluded = ("is_first", "is_last", "is_terminal", "height_map")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.image_shapes = {k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)}
        self.vector_shapes = {k: v for k, v in shapes.items() if len(v) in (1, 2) and re.match(mlp_keys, k)}

        if self.image_shapes:
            some_shape = next(iter(self.image_shapes.values()))
            shape = (sum(x[0] for x in self.image_shapes.values()), *some_shape[1:])
            self.image_decoder = ConvDecoder(
                feat_size,
                shape,
                cnn_depth,
                act,
                norm,
                kernel_size,
                minres,
                outscale=outscale,
                cnn_sigmoid=cnn_sigmoid,
            )
        if self.vector_shapes:
            self.vector_decoder = MLP(
                feat_size,
                self.vector_shapes,
                mlp_layers,
                mlp_units,
                act,
                norm,
                vector_dist,
                outscale=outscale,
                name="Decoder",
            )
        self._image_dist = image_dist

    def forward(self, features: torch.Tensor) -> dict[str, Any]:
        dists: dict[str, Any] = {}
        if self.image_shapes:
            outputs = self.image_decoder(features)
            split_sizes = [v[0] for v in self.image_shapes.values()]
            outputs = torch.split(outputs, split_sizes, -3)
            dists.update({key: self._make_image_dist(o) for key, o in zip(self.image_shapes.keys(), outputs)})
        if self.vector_shapes:
            dists.update(self.vector_decoder(features))
        return dists

    def _make_image_dist(self, mean: torch.Tensor) -> Any:
        if self._image_dist == "normal":
            return ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, 1), 3))
        if self._image_dist == "mse":
            return MSEDist(mean)
        raise NotImplementedError(self._image_dist)
