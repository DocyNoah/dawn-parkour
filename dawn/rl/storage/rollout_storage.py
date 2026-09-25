from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import torch

from dawn.rl.utils import split_and_pad_trajectories

if TYPE_CHECKING:
    from collections.abc import Generator


MiniBatchType: TypeAlias = tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    None,
    None,
    None,
]
RecurrentMiniBatchType: TypeAlias = tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | list[torch.Tensor],
    torch.Tensor | list[torch.Tensor],
    torch.Tensor,
]


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.obs_dict = {}
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states_a = None
            self.hidden_states_c = None

        def clear(self) -> None:
            self.__init__()

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape_dict: dict[str, list[int]],
        actions_shape: tuple[int, ...],
        device: str = "cpu",
    ):
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.obs_shape_dict = obs_shape_dict
        self.actions_shape = actions_shape

        self.obs_dict = {
            k: torch.zeros(num_transitions_per_env, num_envs, *v, device=self.device) for k, v in obs_shape_dict.items()
        }

        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def add_transitions(self, transition: Transition) -> None:
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        for k, v in transition.obs_dict.items():
            if k not in self.obs_dict:
                continue
            self.obs_dict[k][self.step].copy_(v)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)

        self._save_hidden_states_a(transition.hidden_states_a)
        self._save_hidden_states_c(transition.hidden_states_c)

        self.step += 1

    def _save_hidden_states_a(
        self,
        hidden_states_a: torch.Tensor | None,
    ) -> None:
        if hidden_states_a is None:
            return

        hid_a = hidden_states_a if isinstance(hidden_states_a, tuple) else (hidden_states_a,)

        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.num_transitions_per_env, *h.shape, device=self.device) for h in hid_a
            ]

        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])

    def _save_hidden_states_c(
        self,
        hidden_states_c: torch.Tensor | None,
    ) -> None:
        if hidden_states_c is None:
            return

        hid_c = hidden_states_c if isinstance(hidden_states_c, tuple) else (hidden_states_c,)

        if self.saved_hidden_states_c is None:
            self.saved_hidden_states_c = [
                torch.zeros(self.num_transitions_per_env, *h.shape, device=self.device) for h in hid_c
            ]

        for i in range(len(hid_c)):
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self) -> None:
        self.step = 0

    def compute_returns(
        self,
        last_values: torch.Tensor,
        gamma: float,
        lam: float,
        normalize_advantage: bool = True,
    ) -> None:
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]

            next_is_not_terminal = 1.0 - self.dones[step].float()

            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]

            advantage = delta + next_is_not_terminal * gamma * lam * advantage

            self.returns[step] = advantage + self.values[step]

        self.advantages = self.returns - self.values

        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int = 8,
    ) -> Generator[MiniBatchType, None, None]:
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        obs_dict = {k: v.flatten(0, 1) for k, v in self.obs_dict.items()}

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch_dict = {k: v[batch_idx] for k, v in obs_dict.items()}
                actions_batch = actions[batch_idx]

                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                yield (
                    obs_batch_dict,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    None,
                    None,
                    None,
                )

    def recurrent_mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int = 8,
    ) -> Generator[RecurrentMiniBatchType, None, None]:
        padded_obs_trajectories_dict = {}
        for k, v in self.obs_dict.items():
            padded_obs_trajectories_dict[k], trajectory_masks = split_and_pad_trajectories(v, self.dones)

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch_dict = {k: v[:, first_traj:last_traj] for k, v in padded_obs_trajectories_dict.items()}
                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                last_was_done = last_was_done.permute(1, 0)

                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]

                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch

                if self.saved_hidden_states_c is not None:
                    hid_c_batch = [
                        saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                        .transpose(1, 0)
                        .contiguous()
                        for saved_hidden_states in self.saved_hidden_states_c
                    ]

                    hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch
                else:
                    hid_c_batch = None

                yield (
                    obs_batch_dict,
                    actions_batch,
                    values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    hid_a_batch,
                    hid_c_batch,
                    masks_batch,
                )

                first_traj = last_traj
