from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.distributions import Normal

from dawn.rl.networks import networks_registry

if TYPE_CHECKING:
    from dawn.rl.configs.train_cfg import ActorCriticPolicyCfg
    from dawn.rl.networks.base import BaseNetwork


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dict: dict[str, torch.Tensor],
        num_actions: int,
        use_critic: bool,
        policy_cfg: ActorCriticPolicyCfg,
        fixed_std: bool = False,
        **kwargs,
    ):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(list(kwargs.keys())))
        super().__init__()

        self.is_recurrent = policy_cfg.is_recurrent

        actor_network_cls = networks_registry.get(policy_cfg.actor_network_name)
        self.actor: BaseNetwork = actor_network_cls(
            obs_dict=obs_dict,
            num_actions=num_actions,
            network_cfg=policy_cfg.actor_network_cfg,
            obs_normalization=policy_cfg.actor_obs_normalization,
        )

        if use_critic:
            critic_network_name = policy_cfg.critic_network_name
            critic_network_cfg = policy_cfg.critic_network_cfg
            if critic_network_name is None or critic_network_cfg is None:
                raise ValueError(
                    "When use_critic=True, policy_cfg must have "
                    "both critic_network_name and critic_network_cfg specified"
                )
            critic_network_cls = networks_registry.get(critic_network_name)
            self.critic: BaseNetwork = critic_network_cls(
                obs_dict=obs_dict,
                num_actions=1,
                network_cfg=critic_network_cfg,
                obs_normalization=policy_cfg.critic_obs_normalization,
            )

        self.obs_keys = self._collect_obs_keys(use_critic)

        self.noise_std_type = policy_cfg.noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(policy_cfg.init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(policy_cfg.init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.fixed_std = fixed_std

        self.distribution = None

        Normal.set_default_validate_args = False

    def reset(self, dones: torch.Tensor | None = None, hidden_states: torch.Tensor | None = None) -> None:
        self.actor.reset_hidden_states(dones, hidden_states)
        if hasattr(self, "critic") and self.critic is not None:
            self.critic.reset_hidden_states(dones, hidden_states)

    def _collect_obs_keys(self, use_critic: bool) -> tuple[str, ...]:
        obs_keys = list(self.actor.obs_keys)
        if use_critic and hasattr(self, "critic") and self.critic is not None:
            obs_keys.extend(self.critic.obs_keys)
        return tuple(dict.fromkeys(obs_keys))

    def get_hidden_states_a(self) -> torch.Tensor | None:
        return self.actor.get_hidden_states()

    def get_hidden_states_c(self) -> torch.Tensor | None:
        if hasattr(self, "critic") and self.critic is not None:
            return self.critic.get_hidden_states()

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        self.actor.detach_hidden_states(dones)
        if hasattr(self, "critic") and self.critic is not None:
            self.critic.detach_hidden_states(dones)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("Forward method is not implemented. Use act or act_inference or evaluate instead.")

    def forward_actor(
        self,
        obs_dict: dict[str, torch.Tensor],
        masks: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.is_recurrent:
            out = self.actor(obs_dict, masks, hidden_states)
        else:
            out = self.actor(obs_dict)
        return out

    def forward_critic(
        self,
        obs_dict: dict[str, torch.Tensor],
        masks: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.critic(obs_dict)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(
        self,
        obs_dict: dict[str, torch.Tensor],
        masks: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mean = self.forward_actor(obs_dict, masks, hidden_states)

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = Normal(mean, std)

    def act(
        self,
        obs_dict: dict[str, torch.Tensor],
        masks: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.update_distribution(obs_dict, masks, hidden_states)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(
        self,
        obs_dict: dict[str, torch.Tensor],
        masks: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        actions_mean = self.forward_actor(obs_dict, masks, hidden_states)
        return actions_mean

    def evaluate(
        self,
        obs_dict: dict[str, torch.Tensor],
        masks: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        value = self.forward_critic(obs_dict, masks, hidden_states)
        return value

    def train_mode(self, mode: bool = True) -> None:
        super().train(mode)
        self.actor.train_mode(mode)
        if hasattr(self, "critic") and self.critic is not None:
            self.critic.train_mode(mode)

    def eval_mode(self) -> None:
        self.train_mode(False)

    def save(self, model_dir: str) -> None:
        actor_model_dir = os.path.join(model_dir, "actor")
        os.makedirs(actor_model_dir, exist_ok=True)
        self.actor.save(actor_model_dir)

        if hasattr(self, "critic") and self.critic is not None:
            critic_model_dir = os.path.join(model_dir, "critic")
            os.makedirs(critic_model_dir, exist_ok=True)
            self.critic.save(critic_model_dir)

    def load(self, model_dir: str, actor_sub_dir: str = "actor", critic_sub_dir: str = "critic") -> None:
        actor_model_path = os.path.join(model_dir, actor_sub_dir)
        critic_model_path = os.path.join(model_dir, critic_sub_dir)

        print(f"[INFO]: ========== Start loading {actor_sub_dir} model ==========")
        self.actor.load(actor_model_path, prefix=actor_sub_dir, raise_on_missing=True)
        print(f"[INFO]: ========== Done loading {actor_sub_dir} model ==========")

        if hasattr(self, "critic") and self.critic is not None:
            print(f"[INFO]: ========== Start loading {critic_sub_dir} model ==========")
            self.critic.load(critic_model_path, prefix=critic_sub_dir, raise_on_missing=True)
            print(f"[INFO]: ========== Done loading {critic_sub_dir} model ==========")
