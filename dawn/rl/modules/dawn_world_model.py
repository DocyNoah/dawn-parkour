from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from torch import nn

from dawn.rl.modules.world_model import WorldModel

if TYPE_CHECKING:
    from dawn.rl.configs.world_model_cfg import DAWNWorldModelCfg
    from dawn.rl.networks.dreamer.rssm import LatentTraj


class DAWNWorldModel(WorldModel):
    _contrastive_enabled: bool
    _contrastive_temperature: float
    _contrastive_scale: float
    _projection_head: nn.Module

    def __init__(
        self,
        config: DAWNWorldModelCfg,
        obs_shape: dict[str, tuple[int, ...]],
        device: str | torch.device,
        num_actions: int,
    ) -> None:
        super().__init__(config, obs_shape, device, num_actions)

        rssm_cfg = config.rssm
        contrastive_cfg = config.contrastive

        self._contrastive_enabled = contrastive_cfg.enabled
        if self._contrastive_enabled:
            self._contrastive_temperature = float(contrastive_cfg.temperature)
            self._contrastive_scale = float(contrastive_cfg.contrastive_coef)
            proj_hidden = int(contrastive_cfg.projection_hidden)
            proj_dim = int(contrastive_cfg.projection_dim)
            feat_size = rssm_cfg.dyn_stoch * rssm_cfg.dyn_discrete + rssm_cfg.dyn_deter
            self._projection_head = nn.Sequential(
                nn.Linear(feat_size, proj_hidden),
                nn.LayerNorm(proj_hidden),
                nn.SiLU(),
                nn.Linear(proj_hidden, proj_dim),
            ).to(self.device)
            self._model_opt.add_parameters(self._projection_head.parameters())

    def _target_key(self, target_name: str, chunk: dict[str, torch.Tensor]) -> str:
        if target_name == "image" and "image_clean" in chunk:
            return "image_clean"
        return target_name

    def load(self, model_dir: str) -> None:
        path = os.path.join(model_dir, "world_model.pt")
        saved_state = torch.load(path, map_location=self.device, weights_only=True)

        self.load_state_dict(saved_state, strict=True)

    def _extra_losses(
        self,
        chunk: dict[str, torch.Tensor],
        latent_traj: LatentTraj,
        post_logit: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del post_logit
        if not self._contrastive_enabled or "image_clean" not in chunk:
            return {}

        data_clean = {**chunk, "image": chunk["image_clean"].clone()}
        embed_clean = self.rssm.obs_embedder(data_clean)
        latent_clean, _, _ = self.rssm.observe_traj(embed_clean, chunk["action"], chunk["is_first"])
        z_noisy = self._projection_head(latent_traj.s)
        z_clean = self._projection_head(latent_clean.s)

        contrastive_loss = self._compute_contrastive_loss(z_noisy, z_clean)
        return {"contrastive": self._contrastive_scale * contrastive_loss}

    def _compute_contrastive_loss(self, z_noisy: torch.Tensor, z_clean: torch.Tensor) -> torch.Tensor:
        B, T, D = z_noisy.shape
        z_noisy = z_noisy.reshape(B * T, D)
        z_clean = z_clean.reshape(B * T, D)

        z_noisy = torch.nn.functional.normalize(z_noisy, dim=-1)
        z_clean = torch.nn.functional.normalize(z_clean, dim=-1)

        logits = z_noisy @ z_clean.T / self._contrastive_temperature
        labels = torch.arange(B * T, device=z_noisy.device)
        loss_nc = torch.nn.functional.cross_entropy(logits, labels)
        loss_cn = torch.nn.functional.cross_entropy(logits.T, labels)
        loss = (loss_nc + loss_cn) / 2

        return loss.expand(B, T)
