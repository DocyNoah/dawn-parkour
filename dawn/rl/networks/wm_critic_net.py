from __future__ import annotations

import torch
import torch.nn as nn

from dawn.rl.networks.base import BaseNetwork


class WMCriticNetwork(BaseNetwork):
    obs_keys = ("prop_hist", "scan", "priv", "wm_feature", "command", "action_hist")

    def __init__(
        self,
        obs_dict: dict[str, torch.Tensor],
        num_actions: int,
        network_cfg: dict,
        obs_normalization: bool = False,
    ):
        super().__init__(obs_dict, num_actions, network_cfg, obs_normalization)

        num_prop_obs = obs_dict["prop_hist"].shape[-1]
        num_scan_obs = obs_dict["scan"].shape[-1]
        num_priv_obs = obs_dict["priv"].shape[-1]
        num_wm_feature_obs = obs_dict["wm_feature"].shape[-1]
        num_command_obs = obs_dict["command"].shape[-1]
        num_action_obs = obs_dict["action_hist"].shape[-1]
        num_obs = num_prop_obs + num_scan_obs + num_priv_obs + num_command_obs + num_action_obs

        wm_encoder_hidden_dims = network_cfg.get("wm_encoder_hidden_dims", [64, 64])
        critic_hidden_dims = network_cfg.get("critic_hidden_dims", [512, 256, 128])

        wm_feature_latent_dim = 32

        wm_encoder_layers = self._make_mlp_layers(
            input_dim=num_wm_feature_obs,
            hidden_dim=wm_encoder_hidden_dims,
            activation_fn=self.activation_fn,
        )
        wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[-1], wm_feature_latent_dim))
        wm_feature_encoder = nn.Sequential(*wm_encoder_layers)

        value_layers = self._make_mlp_layers(
            input_dim=num_obs + wm_feature_latent_dim,
            hidden_dim=critic_hidden_dims,
            activation_fn=self.activation_fn,
        )
        value_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        value = nn.Sequential(*value_layers)

        self.module_dict = nn.ModuleDict(
            {
                "wm_feature_encoder": wm_feature_encoder,
                "value": value,
            }
        )

    def get_env_enc_params(self) -> list[torch.nn.Parameter]:
        raise NotImplementedError("The policy does not use vision observations")

    def get_prop_enc_params(self) -> list[torch.nn.Parameter]:
        return list(self.module_dict["wm_feature_encoder"].parameters())

    def get_vision_enc_params(self) -> list[torch.nn.Parameter]:
        return list(self.module_dict["wm_feature_encoder"].parameters())

    def get_base_policy_params(self) -> list[torch.nn.Parameter]:
        return list(self.module_dict["value"].parameters())

    def _forward_impl(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        prop_obs = obs_dict["prop_hist"][:, -1]
        scan_obs = obs_dict["scan"]
        priv_obs = obs_dict["priv"]
        command_obs = obs_dict["command"]
        action = obs_dict["action_hist"][:, -1]
        obs = torch.cat([priv_obs, prop_obs, command_obs, action, scan_obs], dim=-1)
        wm_feature = obs_dict["wm_feature"]

        wm_feature_latent = self.module_dict["wm_feature_encoder"](wm_feature)

        concat_obs = torch.cat([obs, wm_feature_latent], dim=-1)
        value = self.module_dict["value"](concat_obs)
        return value
