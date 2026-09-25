from typing import Any

import torch
from torch import distributions as torchd
from torch import nn

from dawn.rl.networks.dreamer.distributions import OneHotDist
from dawn.rl.networks.dreamer.rssm.decoder import Decoder
from dawn.rl.networks.dreamer.rssm.dynamics_predictor import DynamicsPredictor
from dawn.rl.networks.dreamer.rssm.encoder import Encoder
from dawn.rl.networks.dreamer.rssm.latent import Latent, LatentTraj
from dawn.rl.networks.dreamer.rssm.obs_embedder import ObsEmbedder
from dawn.rl.networks.dreamer.rssm.sequence_model import SequenceModel
from dawn.rl.utils import resolve_nn_activation


class RSSM(nn.Module):
    def __init__(
        self,
        obs_shape: dict[str, tuple[int, ...]],
        embedder_cfg: dict[str, object],
        decoder_cfg: dict[str, object],
        z_dim: int,
        h_dim: int,
        hidden: int,
        z_classes: int,
        num_actions: int,
        norm: bool,
        activation_fn: str,
        unimix_ratio: float,
        device: str,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.hidden = hidden
        self.z_classes = z_classes
        self.num_actions = num_actions
        self.unimix_ratio = unimix_ratio
        self.device = device

        activation = resolve_nn_activation(activation_fn)

        self.obs_embedder = ObsEmbedder(obs_shape, **embedder_cfg)
        self.embed_dim: int = int(self.obs_embedder.outdim)

        self.encoder = Encoder(
            h_dim=h_dim,
            embed_dim=self.embed_dim,
            hidden=hidden,
            z_dim=z_dim,
            z_classes=z_classes,
            norm=norm,
            activation=activation,
        )

        self.sequence_model = SequenceModel(
            z_flat_dim=z_dim * z_classes,
            action_dim=num_actions,
            hidden=hidden,
            h_dim=h_dim,
            norm=norm,
            activation=activation,
        )

        self.dynamics_predictor = DynamicsPredictor(
            h_dim=h_dim,
            hidden=hidden,
            z_dim=z_dim,
            z_classes=z_classes,
            norm=norm,
            activation=activation,
        )

        self.feat_dim: int = z_dim * z_classes + h_dim
        self.decoder = Decoder(self.feat_dim, obs_shape, **decoder_cfg)

        self.h0 = nn.Parameter(
            torch.zeros((1, h_dim), device=torch.device(device)),
            requires_grad=True,
        )

    def init_latent(self, batch_size: int) -> Latent:
        h = torch.tanh(self.h0).repeat(batch_size, 1)
        z = self._sample_z(self._prior_logit(h), sample=False)
        return Latent(h=h, z=z)

    def categorical(self, logit: torch.Tensor) -> Any:
        return torchd.independent.Independent(OneHotDist(logit, unimix_ratio=self.unimix_ratio), 1)

    def observe_step(
        self,
        prev_latent: Latent | None,
        prev_action: torch.Tensor,
        obs_embed: torch.Tensor,
        is_first: torch.Tensor,
        sample: bool = True,
    ) -> Latent:
        prev_latent, prev_action = self._reset_first(prev_latent, prev_action, is_first)
        h = self._recurrent_forward(prev_latent, prev_action)
        post_logit = self._posterior_logit(h, obs_embed)
        z = self._sample_z(post_logit, sample)
        return Latent(h=h, z=z)

    def observe_traj(
        self,
        obs_embed: torch.Tensor,
        action: torch.Tensor,
        is_first: torch.Tensor,
        init_latent: Latent | None = None,
        sample: bool = True,
    ) -> tuple[LatentTraj, torch.Tensor, torch.Tensor]:
        T = obs_embed.shape[1]
        latent = init_latent

        h_list: list[torch.Tensor] = []
        z_list: list[torch.Tensor] = []
        post_logit_list: list[torch.Tensor] = []
        prior_logit_list: list[torch.Tensor] = []

        for t in range(T):
            latent, prev_action_step = self._reset_first(latent, action[:, t], is_first[:, t])
            h = self._recurrent_forward(latent, prev_action_step)
            prior_logit = self._prior_logit(h)
            post_logit = self._posterior_logit(h, obs_embed[:, t])
            z = self._sample_z(post_logit, sample)
            latent = Latent(h=h, z=z)

            h_list.append(h)
            z_list.append(z)
            post_logit_list.append(post_logit)
            prior_logit_list.append(prior_logit)

        latent_traj = LatentTraj(h=torch.stack(h_list, dim=1), z=torch.stack(z_list, dim=1))
        post_logit_traj = torch.stack(post_logit_list, dim=1)
        prior_logit_traj = torch.stack(prior_logit_list, dim=1)
        return latent_traj, post_logit_traj, prior_logit_traj

    def decode(self, latent: Latent | LatentTraj) -> dict[str, Any]:
        return self.decoder(latent.s)

    def kl_loss(
        self,
        post_logit: torch.Tensor,
        prior_logit: torch.Tensor,
        free: float,
        dyn_scale: float,
        rep_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        kld = torchd.kl.kl_divergence
        post = self.categorical(post_logit)
        prior = self.categorical(prior_logit)
        post_sg = self.categorical(post_logit.detach())
        prior_sg = self.categorical(prior_logit.detach())

        rep_loss = value = kld(post, prior_sg)
        dyn_loss = kld(post_sg, prior)
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss
        return loss, value, dyn_loss, rep_loss

    def _recurrent_forward(self, prev_latent: Latent, prev_action: torch.Tensor) -> torch.Tensor:
        prev_z_flat = prev_latent.z.reshape([*prev_latent.z.shape[:-2], self.z_dim * self.z_classes])
        return self.sequence_model(prev_z_flat, prev_action, prev_latent.h)

    def _prior_logit(self, h: torch.Tensor) -> torch.Tensor:
        return self.dynamics_predictor(h)

    def _posterior_logit(self, h: torch.Tensor, x_embed: torch.Tensor) -> torch.Tensor:
        return self.encoder(h, x_embed)

    def _sample_z(self, logit: torch.Tensor, sample: bool) -> torch.Tensor:
        d = self.categorical(logit)
        return d.sample() if sample else d.mode()

    def _reset_first(
        self,
        prev_latent: Latent | None,
        prev_action: torch.Tensor,
        is_first: torch.Tensor,
    ) -> tuple[Latent, torch.Tensor]:
        B = is_first.shape[0]
        if prev_latent is None or torch.sum(is_first) == B:
            init = self.init_latent(B)
            zero_action = torch.zeros((B, self.num_actions), device=self.device)
            return init, zero_action

        if torch.sum(is_first) > 0:
            mask_a = is_first[:, None]
            prev_action = prev_action * (1.0 - mask_a)
            init = self.init_latent(B)
            mask_h = is_first.reshape((B, 1))
            mask_z = is_first.reshape((B, 1, 1))
            new_h = prev_latent.h * (1.0 - mask_h) + init.h * mask_h
            new_z = prev_latent.z * (1.0 - mask_z) + init.z * mask_z
            prev_latent = Latent(h=new_h, z=new_z)
        return prev_latent, prev_action
