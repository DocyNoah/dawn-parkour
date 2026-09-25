from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.optim as optim

from dawn.rl.storage import ReplayBuffer, RolloutStorage
from dawn.rl.utils.utils import Normalizer, unpad_trajectories

if TYPE_CHECKING:
    from dawn.rl.configs.train_cfg import PPOAMPAlgorithmCfg
    from dawn.rl.datasets.motion_loader import AMPLoader
    from dawn.rl.modules import ActorCritic
    from dawn.rl.networks.sub_networks.amp_discriminator import AMPDiscriminator


class PPOAMPAlgorithm:
    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic: ActorCritic,
        discriminator: AMPDiscriminator,
        amp_data: AMPLoader,
        amp_normalizer: Normalizer,
        min_std: torch.Tensor | None,
        device: str,
        algorithm_cfg: PPOAMPAlgorithmCfg,
        **kwargs,
    ):
        if kwargs:
            print(
                "PPOAMPAlgorithm.__init__ got unexpected arguments, which will be ignored: " + str(list(kwargs.keys()))
            )
        self.device = device

        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)

        self.num_learning_epochs = algorithm_cfg.num_learning_epochs

        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_transition = RolloutStorage.Transition()

        self.amp_storage = ReplayBuffer(
            obs_dim=discriminator.input_dim // 2,
            buffer_size=algorithm_cfg.amp_replay_buffer_size,
            device=self.device,
        )

        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)

        params = [
            {"params": self.actor_critic.parameters(), "name": "actor_critic"},
            {"params": self.discriminator.trunk.parameters(), "weight_decay": 10e-4, "name": "amp_trunk"},
            {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 10e-2, "name": "amp_head"},
        ]
        self.optimizer = optim.Adam(params, lr=algorithm_cfg.learning_rate)

        self.storage: RolloutStorage | None = None

        self.transition = RolloutStorage.Transition()

        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer

        self.clip_param = algorithm_cfg.clip_param
        self.num_learning_epochs = algorithm_cfg.num_learning_epochs
        self.num_mini_batches = algorithm_cfg.num_mini_batches
        self.value_loss_coef = algorithm_cfg.value_loss_coef
        self.entropy_coef = algorithm_cfg.entropy_coef
        self.vel_predict_coef = algorithm_cfg.vel_predict_coef
        self.gamma = algorithm_cfg.gamma
        self.lam = algorithm_cfg.lam
        self.max_grad_norm = algorithm_cfg.max_grad_norm
        self.use_clipped_value_loss = algorithm_cfg.use_clipped_value_loss
        self.desired_kl = algorithm_cfg.desired_kl
        self.schedule = algorithm_cfg.schedule
        self.learning_rate = algorithm_cfg.learning_rate
        self.min_std = min_std
        self.normalize_advantage_per_mini_batch = algorithm_cfg.normalize_advantage_per_mini_batch

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape_dict: dict[str, list[int]],
        action_shape: tuple[int, ...],
    ) -> None:
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            obs_shape_dict,
            action_shape,
            self.device,
        )

    def train_mode(self) -> None:
        self.actor_critic.train_mode()

    def act(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states_a = self.actor_critic.get_hidden_states_a()
            self.transition.hidden_states_c = self.actor_critic.get_hidden_states_c()

        self.transition.actions = self.actor_critic.act(obs_dict).detach()
        self.transition.values = self.actor_critic.evaluate(obs_dict).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()

        for k, v in obs_dict.items():
            self.transition.obs_dict[k] = v

        return self.transition.actions

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict,
        amp_obs: torch.Tensor,
    ) -> None:
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if "time_outs" in infos:
            transition_values = getattr(self.transition, "values")
            self.transition.rewards += self.gamma * torch.squeeze(
                transition_values * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        if self.transition.obs_dict["amp_obs"] is not None:
            prev_amp_obs = self.transition.obs_dict["amp_obs"]
        else:
            prev_amp_obs = torch.zeros_like(amp_obs)
        self.amp_storage.insert(prev_amp_obs, amp_obs)

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, obs_dict: dict[str, torch.Tensor]) -> None:
        last_values = self.actor_critic.evaluate(obs_dict).detach()
        self.storage.compute_returns(
            last_values,
            self.gamma,
            self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
        )

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_vel_predict_loss = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        mean_entropy = 0

        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )

        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
            (
                obs_batch_dict,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_a_batch,
                hid_c_batch,
                masks_batch,
            ) = sample

            original_batch_size = actions_batch.shape[0]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            self.actor_critic.act(
                obs_dict=obs_batch_dict,
                masks=masks_batch,
                hidden_states=hid_a_batch,
            )
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)

            value_batch = self.actor_critic.evaluate(
                obs_dict=obs_batch_dict,
                masks=masks_batch,
                hidden_states=hid_c_batch,
            )

            mu_batch = self.actor_critic.action_mean[:original_batch_size]
            sigma_batch = self.actor_critic.action_std[:original_batch_size]
            entropy_batch = self.actor_critic.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                if value_batch.shape != target_values_batch.shape:
                    value_batch = unpad_trajectories(value_batch, masks_batch)
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            predicted_linear_vel = self.actor_critic.actor.get_linear_vel(obs_batch_dict)
            target_linear_vel = obs_batch_dict["priv"][:, -3:]
            vel_predict_loss = (predicted_linear_vel - target_linear_vel).pow(2).mean()

            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert

            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                    expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)

            policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
            expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))

            expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
            policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
            amp_loss = 0.5 * (expert_loss + policy_loss)

            grad_pen_loss = self.discriminator.compute_grad_pen(*sample_amp_expert, lambda_=10)

            loss = (
                surrogate_loss
                + self.vel_predict_coef * vel_predict_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + amp_loss
                + grad_pen_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if not self.actor_critic.fixed_std and self.min_std is not None:
                self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_std)

            if self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state.cpu().numpy())
                self.amp_normalizer.update(expert_state.cpu().numpy())

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_vel_predict_loss += vel_predict_loss.item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()
            mean_entropy += entropy_batch.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_vel_predict_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        mean_entropy /= num_updates

        self.storage.clear()

        return {
            "mean_value_loss": mean_value_loss,
            "mean_surrogate_loss": mean_surrogate_loss,
            "mean_vel_predict_loss": mean_vel_predict_loss,
            "mean_amp_loss": mean_amp_loss,
            "mean_grad_pen_loss": mean_grad_pen_loss,
            "mean_policy_pred": mean_policy_pred,
            "mean_expert_pred": mean_expert_pred,
            "mean_entropy": mean_entropy,
        }
