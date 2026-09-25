from __future__ import annotations

import numpy as np
import torch


class DAWNRolloutStorage:
    # Camera images use compact camera rows; other observations use environment rows.
    _WORLD_MODEL_BATCH_SKIP_KEYS = ("forward_height_map", "prop_clean", "image", "image_clean")

    _WORLD_MODEL_OBS_SKIP_KEYS = ("is_first", "image")

    def __init__(
        self,
        num_envs: int,
        num_actions: int,
        max_episode_length: int | float,
        update_interval: int,
        prop_dim: int,
        depth_shape: tuple[int, ...],
        forward_height_map_dim: int,
        depth_index: np.ndarray,
        camera_buffer_index_by_env: np.ndarray,
    ) -> None:
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.update_interval = update_interval
        self.prop_dim = prop_dim
        self.depth_shape = depth_shape
        self.forward_height_map_dim = forward_height_map_dim
        self.depth_index = depth_index
        self.camera_buffer_index_by_env = camera_buffer_index_by_env

        self.rollout_capacity = int(max_episode_length / update_interval) + 3
        self.completed = self._new_rollout_storage()
        self.active = self._new_rollout_storage()
        self.dataset_size = np.zeros(num_envs)
        self.active_index = np.zeros(num_envs)

    @property
    def total_size(self) -> float:
        return float(np.sum(self.dataset_size))

    def complete_episodes(self, reset_env_ids: np.ndarray) -> float:
        for key, dataset_tensor in self.completed.items():
            if key in ("image", "image_clean"):
                self._copy_camera_entries(key, reset_env_ids, dataset_tensor)
                continue
            dataset_tensor[reset_env_ids, :] = self.active[key][reset_env_ids]

        self.dataset_size[reset_env_ids] = self.active_index[reset_env_ids]
        self.active_index[reset_env_ids] = 0
        return self.total_size

    def store_common_step(
        self,
        env_ids: np.ndarray,
        wm_obs: dict[str, torch.Tensor],
        wm_action: torch.Tensor,
    ) -> None:
        time_index = self.active_index[env_ids]
        for key, value in wm_obs.items():
            if key not in self._WORLD_MODEL_OBS_SKIP_KEYS:
                self.active[key][env_ids, time_index] = value[env_ids].to("cpu")
        self.active["action"][env_ids, time_index] = wm_action[env_ids].to("cpu")
        self.active_index[env_ids] += 1

    def store_real_depth_step(
        self,
        obs_dict: dict[str, torch.Tensor],
        depth_clean: torch.Tensor,
        depth_noisy: torch.Tensor,
    ) -> None:
        depth_index = self.depth_index

        camera_buffer_ids = np.arange(depth_index.shape[0])
        time_index = self.active_index[depth_index]

        self.active["forward_height_map"][
            range(self.num_envs),
            self.active_index,
            :,
        ] = obs_dict["forward_height_map"].to("cpu")

        self.active["image"][camera_buffer_ids, time_index] = depth_noisy[depth_index].to("cpu")
        self.active["image_clean"][camera_buffer_ids, time_index] = depth_clean[depth_index].to("cpu")

    def sample_depth_predictor_batch(
        self,
        batch_size: int,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_idx = np.random.choice(self.depth_index, batch_size, replace=True)
        time_index = [np.random.randint(0, self.dataset_size[idx] + 1) for idx in batch_idx]
        forward_heightmap = self.completed["forward_height_map"][batch_idx, time_index].to(device)
        prop = self.completed["prop_clean"][batch_idx, time_index].to(device)
        camera_buffer_idx = self.camera_buffer_index_by_env[batch_idx]
        depth_clean = self.completed["image_clean"][camera_buffer_idx, time_index].to(device)
        return forward_heightmap, prop, depth_clean

    def sample_world_model_indices(
        self,
        batch_size: int,
        max_batch_length: int,
    ) -> tuple[np.ndarray, int, list[int]] | None:
        p = self.dataset_size / np.sum(self.dataset_size)
        batch_idx = np.random.choice(range(self.num_envs), batch_size, replace=True, p=p)

        batch_length = min(int(self.dataset_size[batch_idx].min()), max_batch_length)
        if batch_length <= 1:
            return None
        batch_end_idx = [np.random.randint(batch_length, self.dataset_size[idx] + 1) for idx in batch_idx]
        return batch_idx, batch_length, batch_end_idx

    def collect_world_model_common_batch(
        self,
        batch_idx: np.ndarray,
        batch_end_idx: list[int],
        batch_length: int,
    ) -> dict[str, torch.Tensor]:
        batch_data: dict[str, torch.Tensor] = {}
        for key, value in self.completed.items():
            if key in self._WORLD_MODEL_BATCH_SKIP_KEYS:
                continue
            sequence_values = []
            for idx, end_idx in zip(batch_idx, batch_end_idx):
                sequence_values.append(value[idx, end_idx - batch_length : end_idx])
            batch_data[key] = torch.stack(sequence_values)
        return batch_data

    def real_depth_sequence(
        self,
        env_idx: int,
        end_idx: int,
        batch_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        camera_buffer_idx = self.camera_buffer_index_by_env[env_idx]
        if camera_buffer_idx < 0:
            return None
        batch_slice = slice(end_idx - batch_length, end_idx)
        return (
            self.completed["image"][camera_buffer_idx, batch_slice],
            self.completed["image_clean"][camera_buffer_idx, batch_slice],
        )

    def prediction_inputs(
        self,
        env_idx: int,
        end_idx: int,
        batch_length: int,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_slice = slice(end_idx - batch_length, end_idx)
        forward_heightmap = self.completed["forward_height_map"][env_idx, batch_slice].to(device)
        prop = self.completed["prop_clean"][env_idx, batch_slice].to(device)
        return forward_heightmap, prop

    def _new_rollout_storage(self) -> dict[str, torch.Tensor]:
        env_shape = (self.num_envs, self.rollout_capacity)
        camera_shape = (self.depth_index.shape[0], self.rollout_capacity, *self.depth_shape)
        storage = {
            "prop": torch.zeros((*env_shape, self.prop_dim), device="cpu"),
            "action": torch.zeros(
                (*env_shape, self.num_actions * self.update_interval),
                device="cpu",
            ),
            "image": torch.zeros(camera_shape, device="cpu"),
            "image_clean": torch.zeros(camera_shape, device="cpu"),
            "prop_clean": torch.zeros((*env_shape, self.prop_dim), device="cpu"),
            "forward_height_map": torch.zeros((*env_shape, self.forward_height_map_dim), device="cpu"),
        }
        return storage

    def _copy_camera_entries(
        self,
        key: str,
        reset_env_ids: np.ndarray,
        dataset_tensor: torch.Tensor,
    ) -> None:
        for env_id in reset_env_ids:
            camera_buffer_idx = self.camera_buffer_index_by_env[env_id]
            if camera_buffer_idx >= 0:
                dataset_tensor[camera_buffer_idx, :] = self.active[key][camera_buffer_idx]
