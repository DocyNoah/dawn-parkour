import torch


def normalize_depth_image(depth_image: torch.Tensor, near_clip: float = 0.28, far_clip: float = 5.0) -> torch.Tensor:
    return (depth_image - near_clip) / (far_clip - near_clip) - 0.5


def clip_depth_image(depth_image: torch.Tensor, near_clip: float = 0.28, far_clip: float = 5.0) -> torch.Tensor:
    return torch.clip(depth_image, near_clip, far_clip)
