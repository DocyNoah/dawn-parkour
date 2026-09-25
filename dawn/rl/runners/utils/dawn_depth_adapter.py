from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    import torch


class DepthNoiseModel(Protocol):
    near_clip: float
    far_clip: float

    def apply(self, clean_depth: torch.Tensor) -> torch.Tensor: ...


class DAWNDepthAdapter:
    def __init__(
        self,
        depth_predictor: torch.nn.Module,
        depth_noise: DepthNoiseModel,
        sequence_depth_noise: DepthNoiseModel,
        depth_index: np.ndarray,
        device: str,
    ) -> None:
        self.depth_predictor = depth_predictor
        self.depth_noise = depth_noise
        self.sequence_depth_noise = sequence_depth_noise
        self.depth_index = depth_index
        self.device = device
        self.reset_live_noise()

    def reset_live_noise(self, env_ids: np.ndarray | None = None) -> None:
        del env_ids

    def _apply_noise(self, clean_depth: torch.Tensor, noise_model: DepthNoiseModel) -> torch.Tensor:
        # Sensor noise operates in meters; the models use normalized depth.
        depth_range = noise_model.far_clip - noise_model.near_clip
        raw_depth = (clean_depth + 0.5) * depth_range + noise_model.near_clip
        noisy_raw = noise_model.apply(raw_depth)
        return (noisy_raw - noise_model.near_clip) / depth_range - 0.5

    def build_live_depth_input(
        self,
        forward_heightmap: torch.Tensor,
        prop_clean: torch.Tensor,
        depth_noisy: torch.Tensor | None,
    ) -> torch.Tensor:
        if depth_noisy is None:
            raise ValueError("DAWN camera runtime requires noisy depth observations.")

        predicted_depth_clean = self.depth_predictor(forward_heightmap, prop_clean)

        predicted_depth_noisy = self._apply_noise(predicted_depth_clean.detach(), self.depth_noise)
        depth_input = predicted_depth_noisy
        depth_input[self.depth_index] = depth_noisy[self.depth_index].to(self.device)
        return depth_input

    def predict_depth_sequences(
        self,
        forward_heightmap: torch.Tensor,
        prop_clean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = forward_heightmap.shape[:2]

        predicted_clean = self.depth_predictor(forward_heightmap.flatten(0, 1), prop_clean.flatten(0, 1))

        predicted_clean = predicted_clean.reshape(batch_size, seq_len, *predicted_clean.shape[1:]).detach()

        predicted_noisy = self._apply_sequence_noise(predicted_clean)
        return predicted_noisy.cpu(), predicted_clean.cpu()

    def _apply_sequence_noise(self, clean_sequence: torch.Tensor) -> torch.Tensor:
        flat_noisy = self._apply_noise(clean_sequence.flatten(0, 1), self.sequence_depth_noise)
        return flat_noisy.reshape(clean_sequence.shape)
