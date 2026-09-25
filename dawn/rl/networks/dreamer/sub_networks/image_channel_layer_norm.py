import torch
from torch import nn


class ImgChLayerNorm(nn.Module):
    def __init__(self, ch: int, eps: float = 1e-03) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
