from __future__ import annotations

from typing import overload

import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from kornia.filters.sobel import spatial_gradient

from dawn.sim.utils.image_processing import clip_depth_image


class DepthNoise:
    def __init__(self, near_clip: float, far_clip: float) -> None:
        self.near_clip = near_clip
        self.far_clip = far_clip

        # Reused masks preserve temporal coherence between refreshes.
        self._edge_dropout_cache: torch.Tensor | None = None
        self._edge_dropout_step: int = 0

        self._far_particle_cache: torch.Tensor | None = None
        self._far_particle_depth_cache: torch.Tensor | None = None
        self._far_particle_step: int = 0

    @overload
    def _normalized_depth_to_raw(self, depth: float) -> float: ...

    @overload
    def _normalized_depth_to_raw(self, depth: torch.Tensor) -> torch.Tensor: ...

    def _normalized_depth_to_raw(self, depth: float | torch.Tensor) -> float | torch.Tensor:
        return self.near_clip + (depth + 0.5) * (self.far_clip - self.near_clip)

    def _normalized_delta_to_raw(self, delta: float) -> float:
        return delta * (self.far_clip - self.near_clip)

    def _edge_dropout(
        self,
        depth_image: torch.Tensor,
        threshold: float = 0.04,
        saturation_point: float = 0.20,
        max_prob: float = 0.60,
        refresh_interval: int = 5,
    ) -> torch.Tensor:
        B, D, H, W = depth_image.shape

        gradients = spatial_gradient(depth_image)
        gradient_norm = torch.norm(gradients, dim=-3).reshape(B, D, H, W)

        threshold_raw = self._normalized_delta_to_raw(threshold)
        saturation_point_raw = self._normalized_delta_to_raw(saturation_point)
        grad_range = saturation_point_raw - threshold_raw
        normalized_grad = ((gradient_norm - threshold_raw) / grad_range).clamp(0.0, 1.0)
        dropout_prob = max_prob * normalized_grad * normalized_grad
        dropout_prob[gradient_norm < threshold_raw] = 0.0

        shape_changed = self._edge_dropout_cache is None or self._edge_dropout_cache.shape != (B, D, H, W)
        if shape_changed or self._edge_dropout_step % refresh_interval == 0:
            self._edge_dropout_cache = torch.rand(B, D, H, W, device=depth_image.device)
        self._edge_dropout_step = (self._edge_dropout_step + 1) % refresh_interval

        dropout_mask = self._edge_dropout_cache < dropout_prob

        dilated = F.max_pool2d(dropout_mask.float(), kernel_size=3, stride=1, padding=1) > 0.5
        edge_region = gradient_norm > threshold_raw
        fill_candidates = dilated & edge_region & (~dropout_mask)
        fill_prob = 0.5 * normalized_grad * normalized_grad
        fill_mask = fill_candidates & (self._edge_dropout_cache < fill_prob)
        dropout_mask = dropout_mask | fill_mask

        depth_image = depth_image.clone()
        depth_image[dropout_mask] = self.near_clip

        return depth_image

    def _far_particle_noise(
        self,
        depth_image: torch.Tensor,
        particle_rate: float = 0.003,
        far_threshold: float = 0.3,
        refresh_interval: int = 10,
    ) -> torch.Tensor:
        B, D, H, W = depth_image.shape

        shape_changed = self._far_particle_cache is None or self._far_particle_cache.shape != (B, D, H, W)
        if shape_changed or self._far_particle_step % refresh_interval == 0:
            self._far_particle_cache = torch.rand(B, D, H, W, device=depth_image.device)
            particle_depth = torch.rand(B, D, H, W, device=depth_image.device) * 0.7 - 0.3
            self._far_particle_depth_cache = self._normalized_depth_to_raw(particle_depth)
        self._far_particle_step = (self._far_particle_step + 1) % refresh_interval

        far_threshold_raw = self._normalized_depth_to_raw(far_threshold)
        distance_ratio = ((depth_image - far_threshold_raw) / (self.far_clip - far_threshold_raw)).clamp(0.0, 1.0)
        particle_prob = particle_rate * distance_ratio

        particle_mask = self._far_particle_cache < particle_prob
        depth_image = depth_image.clone()
        depth_image[particle_mask] = self._far_particle_depth_cache[particle_mask]

        return depth_image

    def _down_sample(self, img: torch.Tensor) -> torch.Tensor:
        B, D, H, W = img.shape
        img = img.view(B, D, H // 2, 2, W // 2, 2)
        img = img.min(dim=5)[0].min(dim=3)[0]
        img = img.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        return img

    def _gaussian_blur(
        self,
        depth_image: torch.Tensor,
        kernel_size: tuple[int, int] = (3, 3),
        sigma: tuple[float, float] = (1.5, 1.5),
    ) -> torch.Tensor:
        B, D, H, W = depth_image.shape
        return gaussian_blur2d(depth_image, kernel_size, sigma).reshape(B, D, H, W)

    def apply(self, clean_depth: torch.Tensor) -> torch.Tensor:
        near_clip = self.near_clip
        far_clip = self.far_clip

        depth_raw = clean_depth.clone()

        depth_raw = clip_depth_image(depth_raw, near_clip, far_clip)
        sensor_noise = torch.randn_like(depth_raw) * 0.01
        depth_raw = depth_raw + sensor_noise

        depth_raw = clip_depth_image(depth_raw, near_clip, far_clip)

        depth_noisy = self._edge_dropout(depth_raw)

        depth_noisy = self._far_particle_noise(depth_noisy)

        depth_noisy = self._down_sample(depth_noisy)
        depth_noisy = self._gaussian_blur(depth_noisy, kernel_size=(3, 3), sigma=(1.5, 1.5))
        depth_noisy = depth_noisy.clamp(near_clip, far_clip)

        depth_noisy = torch.nan_to_num(depth_noisy, nan=self._normalized_depth_to_raw(0.0))

        return depth_noisy
