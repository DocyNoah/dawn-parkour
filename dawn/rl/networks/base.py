from __future__ import annotations

import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from dawn.rl.utils import make_mlp_layers, resolve_nn_activation


class BaseNetwork(nn.Module, ABC):
    obs_keys: tuple[str, ...] = ()
    rnn: nn.Module

    def __init__(
        self,
        obs_dict: dict[str, torch.Tensor],
        num_actions: int,
        network_cfg: dict,
        obs_normalization: bool = False,
    ):
        super().__init__()

        self.activation_fn = self._resolve_nn_activation(network_cfg.get("activation", "elu"))
        self.module_dict: nn.ModuleDict = None

        self.obs_normalization = obs_normalization
        if self.obs_normalization:
            from dawn.rl.modules import EmpiricalNormalization

            obs_shape_dict = {k: v.shape[1:] for k, v in obs_dict.items()}
            self.obs_normalizer_dict = torch.nn.ModuleDict(
                {k: EmpiricalNormalization(shape=obs_shape_dict[k], until=1.0e8) for k in self.obs_keys}
            )
        else:
            self.obs_normalizer_dict = None

        if self.obs_normalization:
            self.forward = self._forward_with_normalization
        else:
            self.forward = self._forward_without_normalization

    @abstractmethod
    def get_prop_enc_params(self) -> list[torch.nn.Parameter]:
        pass

    @abstractmethod
    def get_env_enc_params(self) -> list[torch.nn.Parameter]:
        pass

    @abstractmethod
    def get_vision_enc_params(self) -> list[torch.nn.Parameter]:
        pass

    @abstractmethod
    def get_base_policy_params(self) -> list[torch.nn.Parameter]:
        pass

    def reset_hidden_states(
        self,
        dones: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> None:
        pass

    def get_hidden_states(self) -> torch.Tensor | None:
        pass

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        pass

    def _normalize_obs(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        obs_dict = {k: v(obs_dict[k]) for k, v in self.obs_normalizer_dict.items()}
        return obs_dict

    def _forward_with_normalization(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        obs_dict = self._normalize_obs(obs_dict)
        return self._forward_impl(obs_dict)

    def _forward_without_normalization(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._forward_impl(obs_dict)

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("Forward implementation required")

    @abstractmethod
    def _forward_impl(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    def _make_mlp_layers(
        self,
        input_dim: int,
        hidden_dim: list[int],
        activation_fn: nn.Module,
    ) -> list[nn.Module]:
        layers = make_mlp_layers(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            activation_fn=activation_fn,
        )
        return layers

    def _resolve_nn_activation(self, activation_name: str) -> nn.Module:
        return resolve_nn_activation(activation_name)

    def save(self, model_dir: str) -> None:
        for name, module in self.module_dict.items():
            model_path = os.path.join(model_dir, f"{name}.pt")
            torch.save(module.state_dict(), model_path)

        if self.obs_normalization and self.obs_normalizer_dict is not None:
            for k, obs_normalizer in self.obs_normalizer_dict.items():
                normalizer_name = f"{k}_obs_normalizer"
                torch.save(
                    obs_normalizer.state_dict(),
                    os.path.join(model_dir, f"{normalizer_name}.pt"),
                )

    def load(self, model_dir: str, prefix: str, raise_on_missing: bool = False) -> None:
        prefix = prefix.rjust(10)
        for name, module in self.module_dict.items():
            model_file = f"{name}.pt"
            model_path = os.path.join(model_dir, model_file)

            if not os.path.exists(model_path):
                msg = "Could not find the model".ljust(25)
                if raise_on_missing:
                    raise FileNotFoundError(
                        f"Required model file not found: {model_path} for {prefix.strip()}'s {model_file}"
                    )
            else:
                try:
                    saved_dict = torch.load(model_path, weights_only=True)
                    module.load_state_dict(saved_dict)
                    msg = "Successfully loaded the model".ljust(25)
                except Exception as exc:
                    msg = "Failed to load the model".ljust(25)
                    if raise_on_missing:
                        raise RuntimeError(
                            f"Required model file failed to load: {model_path} for {prefix.strip()}'s {model_file}"
                        ) from exc

            model_name = f"`{model_file}`".ljust(30)
            print(f"[INFO]: {msg} {prefix}'s {model_name} {'from'.ljust(4)} `{model_dir}`")

        if self.obs_normalization and self.obs_normalizer_dict is not None:
            for key, obs_normalizer in self.obs_normalizer_dict.items():
                normalizer_file = f"{key}_obs_normalizer.pt"
                normalizer_path = os.path.join(model_dir, normalizer_file)

                if not os.path.exists(normalizer_path):
                    msg = "Could not find the model".ljust(25)
                else:
                    try:
                        saved_dict = torch.load(normalizer_path, weights_only=True)
                        obs_normalizer.load_state_dict(saved_dict)
                        msg = "Successfully loaded the model".ljust(25)
                    except Exception:
                        msg = "Failed to load the model".ljust(25)

                normalizer_name = f"`{normalizer_file}`".ljust(30)
                print(f"[INFO]: {msg} {prefix}'s {normalizer_name} {'from'.ljust(4)} `{model_dir}`")

    def train_mode(self, mode: bool = True) -> None:
        super().train(mode)

        if self.obs_normalization and self.obs_normalizer_dict is not None:
            for normalizer in self.obs_normalizer_dict.values():
                if mode:
                    normalizer.train()
                else:
                    normalizer.eval()

        if self.module_dict is not None:
            for module in self.module_dict.values():
                if mode:
                    module.train()
                else:
                    module.eval()

    def eval_mode(self) -> None:
        self.train_mode(False)
