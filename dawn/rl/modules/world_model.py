# MIT License
#
# Copyright (c) 2023 NM512
# Copyright (year) Bytedance Ltd. and/or its affiliates.


import os

import numpy as np
import torch
from torch import nn

from dawn.rl.algorithms.world_model_optimizer import Optimizer, RequiresGrad
from dawn.rl.configs.world_model_cfg import WorldModelCfg
from dawn.rl.networks.dreamer.rssm import RSSM, LatentTraj


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


class WorldModel(nn.Module):
    _RECONSTRUCTION_TARGETS: tuple[str, str] = ("image", "prop")

    _config: WorldModelCfg
    device: str
    num_actions: int
    _use_amp: bool
    embed_size: int
    _model_opt: Optimizer
    _scales: dict[str, float]

    def __init__(
        self,
        config: WorldModelCfg,
        obs_shape: dict[str, tuple[int, ...]],
        device: str | torch.device,
        num_actions: int,
    ) -> None:
        super().__init__()
        self._config = config
        self.device = str(device)
        self.num_actions = num_actions
        self._use_amp = config.rssm.precision == 16

        rssm_cfg = config.rssm
        optim_cfg = config.optim
        encoder_cfg = config.encoder.to_dict()
        decoder_cfg = config.decoder.to_dict()

        self.rssm = RSSM(
            obs_shape=obs_shape,
            embedder_cfg=encoder_cfg,
            decoder_cfg=decoder_cfg,
            z_dim=rssm_cfg.dyn_stoch,
            h_dim=rssm_cfg.dyn_deter,
            hidden=rssm_cfg.dyn_hidden,
            z_classes=rssm_cfg.dyn_discrete,
            num_actions=self.num_actions,
            norm=rssm_cfg.norm,
            activation_fn=rssm_cfg.act,
            unimix_ratio=rssm_cfg.unimix_ratio,
            device=self.device,
        )
        self.embed_size = self.rssm.embed_dim

        self._model_opt = Optimizer(
            "model",
            self.parameters(),
            optim_cfg.model_lr,
            optim_cfg.opt_eps,
            optim_cfg.grad_clip,
            optim_cfg.weight_decay,
            use_amp=self._use_amp,
        )
        self._scales = {
            "image": 1.0,
            "prop": 1.0,
        }

    def _train(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[LatentTraj, dict[str, torch.Tensor], dict[str, int | float]]:
        B = next(iter(data.values())).shape[0]
        micro_B = self._config.algorithm.micro_batch_size
        num_chunks = max(1, B // micro_B)

        self._model_opt.zero_grad()
        total_loss_value = 0.0
        agg_metrics: dict[str, float] = {}

        with RequiresGrad(self):
            for chunk_idx in range(num_chunks):
                start = chunk_idx * micro_B
                end = min(start + micro_B, B)
                chunk = {k: v[start:end] for k, v in data.items()}
                chunk = self.preprocess(chunk)

                with torch.amp.autocast("cuda", enabled=self._use_amp):
                    embed = self.rssm.obs_embedder(chunk)

                    latent_traj, post_logit, prior_logit = self.rssm.observe_traj(
                        embed, chunk["action"], chunk["is_first"]
                    )

                    kl_free = self._config.rssm.kl_free
                    dyn_scale = self._config.rssm.dyn_scale
                    rep_scale = self._config.rssm.rep_scale
                    kl_loss, kl_value, dyn_loss, rep_loss = self.rssm.kl_loss(
                        post_logit, prior_logit, kl_free, dyn_scale, rep_scale
                    )

                    feat = latent_traj.s
                    preds = dict(self.rssm.decoder(feat))

                    losses = {
                        name: -preds[name].log_prob(chunk[self._target_key(name, chunk)])
                        for name in self._RECONSTRUCTION_TARGETS
                    }
                    scaled = {k: v * self._scales.get(k, 1.0) for k, v in losses.items()}
                    model_loss = sum(scaled.values()) + kl_loss

                    extra_losses = self._extra_losses(chunk=chunk, latent_traj=latent_traj, post_logit=post_logit)
                    for extra in extra_losses.values():
                        model_loss = model_loss + extra

                    chunk_loss = torch.mean(model_loss) / num_chunks

                self._model_opt.backward(chunk_loss)
                total_loss_value += chunk_loss.detach().cpu().numpy()

                with torch.no_grad():
                    for name, loss in losses.items():
                        agg_metrics[f"{name}_loss"] = (
                            agg_metrics.get(f"{name}_loss", 0.0) + float(_to_np(loss).mean()) / num_chunks
                        )
                    for name, extra in extra_losses.items():
                        agg_metrics[f"{name}_loss"] = (
                            agg_metrics.get(f"{name}_loss", 0.0) + float(_to_np(torch.mean(extra))) / num_chunks
                        )
                    agg_metrics["dyn_loss"] = (
                        agg_metrics.get("dyn_loss", 0.0) + float(_to_np(dyn_loss).mean()) / num_chunks
                    )
                    agg_metrics["rep_loss"] = (
                        agg_metrics.get("rep_loss", 0.0) + float(_to_np(rep_loss).mean()) / num_chunks
                    )
                    agg_metrics["kl"] = agg_metrics.get("kl", 0.0) + float(_to_np(torch.mean(kl_value))) / num_chunks
                    with torch.amp.autocast("cuda", enabled=self._use_amp):
                        agg_metrics["prior_ent"] = (
                            agg_metrics.get("prior_ent", 0.0)
                            + float(_to_np(torch.mean(self.rssm.categorical(prior_logit).entropy()))) / num_chunks
                        )
                        agg_metrics["post_ent"] = (
                            agg_metrics.get("post_ent", 0.0)
                            + float(_to_np(torch.mean(self.rssm.categorical(post_logit).entropy()))) / num_chunks
                        )

            metrics: dict[str, int | float] = self._model_opt.step(total_loss_value, self.parameters())

        metrics.update(agg_metrics)
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        del end

        with torch.amp.autocast("cuda", enabled=self._use_amp):
            context: dict[str, torch.Tensor] = {
                "embed": embed,
                "feat": latent_traj.s,
                "kl": kl_value,
                "postent": self.rssm.categorical(post_logit).entropy(),
            }
        latent_traj_detached = LatentTraj(h=latent_traj.h.detach(), z=latent_traj.z.detach())
        return latent_traj_detached, context, metrics

    def _target_key(self, target_name: str, chunk: dict[str, torch.Tensor]) -> str:
        del chunk
        return target_name

    def _extra_losses(
        self,
        chunk: dict[str, torch.Tensor],
        latent_traj: LatentTraj,
        post_logit: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del chunk, latent_traj, post_logit
        return {}

    def preprocess(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert "is_first" in obs
        return {k: torch.Tensor(v).to(self.device) for k, v in obs.items()}

    def save(self, model_dir: str) -> None:
        torch.save(self.state_dict(), os.path.join(model_dir, "world_model.pt"))

    def load(self, model_dir: str) -> None:
        path = os.path.join(model_dir, "world_model.pt")
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
