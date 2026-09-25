from __future__ import annotations

import torch
import torch.nn as nn

from dawn.rl.networks.base import BaseNetwork


class WMActorNetwork(BaseNetwork):
    obs_keys = (
        "prop_hist",
        "action_hist",
        "wm_feature",
        "command",
    )

    def __init__(
        self,
        obs_dict: dict[str, torch.Tensor],
        num_actions: int,
        network_cfg: dict,
        obs_normalization: bool = False,
    ):
        super().__init__(obs_dict, num_actions, network_cfg, obs_normalization)

        num_prop_obs = obs_dict["prop_hist"].shape[-1]
        num_wm_feature_obs = obs_dict["wm_feature"].shape[-1]
        num_command_obs = obs_dict["command"].shape[-1]

        wm_encoder_hidden_dim = network_cfg.get("wm_encoder_hidden_dim", [64, 64])
        history_hidden_dim = network_cfg.get("history_hidden_dim", [256, 128])
        base_policy_hidden_dim = network_cfg.get("base_policy_hidden_dim", [256, 128, 64])

        latent_dim = 35
        wm_latent_dim = 32

        wm_encoder_layers = self._make_mlp_layers(
            input_dim=num_wm_feature_obs,
            hidden_dim=wm_encoder_hidden_dim,
            activation_fn=self.activation_fn,
        )
        wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dim[-1], wm_latent_dim))
        wm_feature_encoder = nn.Sequential(*wm_encoder_layers)

        num_history_obs = (
            obs_dict["prop_hist"].shape[-2] * num_prop_obs + obs_dict["action_hist"].shape[-2] * num_actions
        )

        history_layers = self._make_mlp_layers(
            input_dim=num_history_obs,
            hidden_dim=history_hidden_dim,
            activation_fn=self.activation_fn,
        )
        history_layers.append(nn.Linear(history_hidden_dim[-1], latent_dim))
        history_encoder = nn.Sequential(*history_layers)

        input_dim = num_command_obs + wm_latent_dim + latent_dim

        base_policy_layers = self._make_mlp_layers(
            input_dim=input_dim,
            hidden_dim=base_policy_hidden_dim,
            activation_fn=self.activation_fn,
        )
        base_policy_layers.append(nn.Linear(base_policy_hidden_dim[-1], num_actions))
        base_policy = nn.Sequential(*base_policy_layers)

        self.module_dict = nn.ModuleDict(
            {
                "wm_feature_encoder": wm_feature_encoder,
                "history_encoder": history_encoder,
                "base_policy": base_policy,
            }
        )

    def get_prop_enc_params(self) -> list[torch.nn.Parameter]:
        return list(self.module_dict["history_encoder"].parameters())

    def get_env_enc_params(self) -> list[torch.nn.Parameter]:
        raise NotImplementedError("The policy does not use vision observations")

    def get_vision_enc_params(self) -> list[torch.nn.Parameter]:
        return list(self.module_dict["wm_feature_encoder"].parameters())

    def get_base_policy_params(self) -> list[torch.nn.Parameter]:
        return list(self.module_dict["base_policy"].parameters())

    def get_linear_vel(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        prop_hist_obs = obs_dict["prop_hist"]
        action_hist_obs = obs_dict["action_hist"]
        prop_action_hist_obs = torch.cat([prop_hist_obs, action_hist_obs], dim=-1)
        flatten_prop_action_hist_obs = prop_action_hist_obs.flatten(start_dim=-2)
        history_latent = self.module_dict["history_encoder"](flatten_prop_action_hist_obs)
        linear_vel = history_latent[:, -3:]
        return linear_vel

    def _forward_impl(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        prop_hist_obs = obs_dict["prop_hist"]
        action_hist_obs = obs_dict["action_hist"]
        prop_action_hist_obs = torch.cat([prop_hist_obs, action_hist_obs], dim=-1)
        flatten_prop_action_hist_obs = prop_action_hist_obs.flatten(start_dim=-2)

        history_latent = self.module_dict["history_encoder"](flatten_prop_action_hist_obs)

        wm_latent = self.module_dict["wm_feature_encoder"](obs_dict["wm_feature"])
        command_obs = obs_dict["command"]

        obs = torch.cat([history_latent, command_obs, wm_latent], dim=-1)
        output = self.module_dict["base_policy"](obs)
        return output
