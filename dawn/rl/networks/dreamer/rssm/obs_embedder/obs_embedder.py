import re

import torch
from torch import nn

from dawn.rl.networks.dreamer.mlp import MLP
from dawn.rl.networks.dreamer.rssm.obs_embedder.conv_obs_encoder import ConvObsEncoder


class ObsEmbedder(nn.Module):
    def __init__(
        self,
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
        symlog_inputs: bool,
    ) -> None:
        super().__init__()

        excluded = ("is_first", "is_last", "is_terminal", "reward", "height_map")
        shapes = {k: v for k, v in shapes.items() if k not in excluded and not k.startswith("log_")}

        self.image_shapes = {k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)}
        self.vector_shapes = {k: v for k, v in shapes.items() if len(v) in (1, 2) and re.match(mlp_keys, k)}

        self.outdim = 0
        if self.image_shapes:
            input_ch = sum(v[0] for v in self.image_shapes.values())
            _, height, width = next(iter(self.image_shapes.values()))
            input_shape = (input_ch, height, width)
            self.image_embedder = ConvObsEncoder(input_shape, cnn_depth, act, norm, kernel_size, minres)
            self.outdim += self.image_embedder.outdim
        if self.vector_shapes:
            input_size = sum(sum(v) for v in self.vector_shapes.values())
            self.vector_embedder = MLP(
                input_size,
                None,
                mlp_layers,
                mlp_units,
                act,
                norm,
                symlog_inputs=symlog_inputs,
                name="Encoder",
            )
            self.outdim += mlp_units

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        if self.image_shapes:
            inputs = torch.cat([obs[k] for k in self.image_shapes], -3)
            outputs.append(self.image_embedder(inputs))
        if self.vector_shapes:
            inputs = torch.cat([obs[k] for k in self.vector_shapes], -1)
            outputs.append(self.vector_embedder(inputs))
        return torch.cat(outputs, -1)
