from __future__ import annotations

import importlib
import os
import pathlib
from typing import TYPE_CHECKING

import git
import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable


def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    if act_name.lower() == "elu":
        return torch.nn.ELU()
    elif act_name.lower() == "selu":
        return torch.nn.SELU()
    elif act_name.lower() == "relu":
        return torch.nn.ReLU()
    elif act_name.lower() == "crelu":
        return torch.nn.CELU()
    elif act_name.lower() == "lrelu":
        return torch.nn.LeakyReLU()
    elif act_name.lower() == "tanh":
        return torch.nn.Tanh()
    elif act_name.lower() == "sigmoid":
        return torch.nn.Sigmoid()
    elif act_name.lower() == "silu":
        return torch.nn.SiLU()
    elif act_name.lower() == "identity":
        return torch.nn.Identity()
    else:
        raise ValueError(f"Invalid activation function '{act_name}'.")


def split_and_pad_trajectories(tensor: torch.Tensor, dones: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dones = dones.clone()
    dones[-1] = 1

    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))

    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()

    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)

    dummy_trajectory = torch.zeros(tensor.shape[0], *tensor.shape[2:], device=tensor.device)

    trajectories = (
        *trajectories,
        dummy_trajectory,
    )

    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)

    padded_trajectories = padded_trajectories[:, :-1]

    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories: torch.Tensor, masks: torch.Tensor | None) -> torch.Tensor:
    if masks is None:
        return trajectories

    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, trajectories.shape[0], trajectories.shape[-1])
        .transpose(1, 0)
    )


def store_code_state(logdir: str, repositories: list[str]) -> list[str]:
    git_log_dir = os.path.join(logdir, "git")
    os.makedirs(git_log_dir, exist_ok=True)
    file_paths = []
    for repository_file_path in repositories:
        try:
            repo = git.Repo(repository_file_path, search_parent_directories=True)
        except Exception:
            print(f"Could not find git repository in {repository_file_path}. Skipping.")

            continue

        repo_name = pathlib.Path(repo.working_dir).name
        t = repo.head.commit.tree

        diff_file_name = os.path.join(git_log_dir, f"{repo_name}.diff")

        if not os.path.isfile(diff_file_name):
            print(f"Storing git diff for '{repo_name}' in: {diff_file_name}")
            with open(diff_file_name, "x", encoding="utf-8") as f:
                content = f"--- git status ---\n{repo.git.status()} \n\n\n--- git diff ---\n{repo.git.diff(t)}"
                f.write(content)

            file_paths.append(diff_file_name)

        commit_file_name = os.path.join(git_log_dir, f"{repo_name}.log")

        if not os.path.isfile(commit_file_name):
            print(f"Storing git commit for '{repo_name}' in: {commit_file_name}")
            with open(commit_file_name, "x", encoding="utf-8") as f:
                f.write("--- recent 5 commits ---\n")
                for commit in list(repo.iter_commits(max_count=5)):
                    commit_date = commit.committed_datetime.strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"{commit_date}\n{commit.summary}\n{commit.author}\n{commit.hexsha[:8]}\n\n")

            file_paths.append(commit_file_name)
    return file_paths


def string_to_callable(name: str) -> Callable:
    try:
        mod_name, attr_name = name.split(":")
        mod = importlib.import_module(mod_name)
        callable_object = getattr(mod, attr_name)

        if callable(callable_object):
            return callable_object
        else:
            raise ValueError(f"The imported object is not callable: '{name}'")
    except AttributeError as e:
        msg = (
            "We could not interpret the entry as a callable object. The format of input should be"
            f" 'module:attribute_name'\nWhile processing input '{name}', received the error:\n {e}."
        )
        raise ValueError(msg)


def make_mlp_layers(
    input_dim: int,
    hidden_dim: list[int],
    activation_fn: torch.nn.Module,
) -> list[torch.nn.Module]:
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim[0]))
    layers.append(activation_fn)
    for layer_index in range(len(hidden_dim) - 1):
        layers.append(nn.Linear(hidden_dim[layer_index], hidden_dim[layer_index + 1]))
        layers.append(activation_fn)
    return layers


class RunningMeanStd:
    def __init__(self, eps: float = 1e-4, shape: tuple[int, ...] = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_t = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m_t / tot_count

        new_count = tot_count

        self.mean, self.var, self.count = new_mean, new_var, new_count


class Normalizer(RunningMeanStd):
    def __init__(self, input_dim: int, eps: float = 1e-4, clip_obs: float = 10.0):
        super().__init__(shape=input_dim)
        self.eps = eps
        self.clip_obs = clip_obs

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + self.eps), -self.clip_obs, self.clip_obs)

    def normalize_torch(self, x: torch.Tensor, device: str) -> torch.Tensor:
        mean_torch = torch.tensor(self.mean, device=device, dtype=torch.float32)
        std_torch = torch.sqrt(torch.tensor(self.var + self.eps, device=device, dtype=torch.float32))
        return torch.clamp((x - mean_torch) / std_torch, -self.clip_obs, self.clip_obs)


_EPS = np.finfo(float).eps * 4.0


def quaternion_slerp(
    q0: torch.Tensor,
    q1: torch.Tensor,
    fraction: torch.Tensor,
    spin: float = 0.0,
    shortestpath: bool = True,
) -> torch.Tensor:
    out = torch.zeros_like(q0)

    zero_mask = torch.isclose(fraction, torch.zeros_like(fraction)).squeeze()
    ones_mask = torch.isclose(fraction, torch.ones_like(fraction)).squeeze()
    out[zero_mask] = q0[zero_mask]
    out[ones_mask] = q1[ones_mask]

    d = torch.sum(q0 * q1, dim=-1, keepdim=True)
    dist_mask = (torch.abs(torch.abs(d) - 1.0) < _EPS).squeeze()
    out[dist_mask] = q0[dist_mask]

    if shortestpath:
        d_old = torch.clone(d)
        d = torch.where(d_old < 0, -d, d)
        q1 = torch.where(d_old < 0, -q1, q1)

    angle = torch.acos(d) + spin * torch.pi
    angle_mask = (torch.abs(angle) < _EPS).squeeze()
    out[angle_mask] = q0[angle_mask]

    final_mask = torch.logical_or(zero_mask, ones_mask)
    final_mask = torch.logical_or(final_mask, dist_mask)
    final_mask = torch.logical_or(final_mask, angle_mask)
    final_mask = torch.logical_not(final_mask)

    isin = 1.0 / angle
    q0 *= torch.sin((1.0 - fraction) * angle) * isin
    q1 *= torch.sin(fraction * angle) * isin
    q0 += q1
    out[final_mask] = q0[final_mask]
    return out
