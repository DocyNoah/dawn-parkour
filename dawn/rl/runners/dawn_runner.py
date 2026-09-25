from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.optim as optim

from dawn.rl.algorithms import PPOAMPAlgorithm
from dawn.rl.datasets import AMPLoader
from dawn.rl.logging import (
    DepthPredictorLogger,
    LoggingManager,
    PolicyLogger,
    WorldModelLogger,
)
from dawn.rl.modules import ActorCritic, DAWNWorldModel, DepthPredictor
from dawn.rl.networks.sub_networks import AMPDiscriminator
from dawn.rl.runners.utils import DAWNDepthAdapter
from dawn.rl.runners.utils.world_model_contracts import (
    build_world_model_obs_template,
    infer_world_model_depth_index,
    infer_world_model_observation_spec,
    validate_world_model_observation_keys,
)
from dawn.rl.storage import DAWNRolloutStorage
from dawn.rl.utils import Normalizer, store_code_state
from dawn.sim.utils.depth_noise_model import DepthNoise

if TYPE_CHECKING:
    from collections.abc import Callable

    from dawn.rl.configs.train_cfg import DawnTrainCfg, LoggingCfg
    from dawn.rl.networks.dreamer.rssm import Latent
    from dawn.sim.utils.env_wrappers import VecEnvWrapper


def _mean_scalar(value: np.ndarray | torch.Tensor) -> float:
    if value.ndim > 0:
        return float(np.mean(value))
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


@dataclass
class DawnRolloutState:
    wm_obs: dict[str, torch.Tensor]
    wm_is_first: torch.Tensor
    wm_action_history: torch.Tensor
    wm_latent: Latent | None = None
    wm_action: torch.Tensor | None = None
    global_step: int = 0

    def mark_world_model_updated(self, wm_latent: Latent) -> None:
        self.wm_latent = wm_latent
        self.wm_is_first[:] = 0

    def advance_step(self, actions: torch.Tensor) -> None:
        self.global_step += 1
        self.wm_action_history = torch.cat([self.wm_action_history[:, 1:], actions.unsqueeze(1)], dim=1)
        self.wm_action = self.wm_action_history.flatten(1)

    def reset_finished(self, reset_env_ids: np.ndarray) -> None:
        self.wm_action_history[reset_env_ids, :] = 0
        if self.wm_action is not None:
            self.wm_action = self.wm_action_history.flatten(1)
        self.wm_is_first[reset_env_ids] = 1

    def active_env_ids(self) -> np.ndarray:
        return (1 - self.wm_is_first).nonzero(as_tuple=False).flatten().cpu().numpy()


class DawnRunner:
    def __init__(
        self,
        env: VecEnvWrapper,
        train_cfg: DawnTrainCfg,
        log_dir: str,
        logging_cfg: LoggingCfg,
        device: str = "cpu",
    ):
        self.cfg = train_cfg
        self.runner_cfg = train_cfg.runner
        self.alg_cfg = train_cfg.algorithm
        self.policy_cfg = train_cfg.policy
        self.amp_cfg = train_cfg.amp
        self.wm_cfg = train_cfg.wm
        self.depth_predictor_cfg = train_cfg.depth_predictor
        self.device = device
        self.env = env

        obs_dict, _ = self.env.get_observations()
        self._depth_index = infer_world_model_depth_index(self.env)
        self._obs_spec = infer_world_model_observation_spec(obs_dict)
        self.wm_update_interval = int(self.env.cfg.depth.update_interval)

        wm_obs_shape_dict = {
            "prop": (self._obs_spec.prop_with_command_dim,),
            "image": self._obs_spec.depth_shape,
        }
        print("Building world model...")
        self._world_model = DAWNWorldModel(
            config=self.wm_cfg,
            obs_shape=wm_obs_shape_dict,
            device=self.device,
            num_actions=self.env.num_actions * self.wm_update_interval,
        ).to(self.device)
        print("Finished building world model.")
        self.wm_feature_dim = self.wm_cfg.rssm.dyn_deter
        self._camera_buffer_index_by_env = np.full(self.env.num_envs, -1, dtype=np.int64)
        if self._depth_index.size > 0:
            self._camera_buffer_index_by_env[self._depth_index] = np.arange(self._depth_index.shape[0], dtype=np.int64)

        obs_dict = {**obs_dict, "wm_feature": torch.zeros(self.env.num_envs, self.wm_feature_dim, device=self.device)}

        self.depth_predictor = DepthPredictor(
            forward_heightmap_dim=self._obs_spec.forward_height_map_dim,
            prop_dim=self._obs_spec.prop_with_command_dim,
            depth_image_dims=self._obs_spec.depth_shape[-2:],
            cnn_base_channels=self.depth_predictor_cfg.cnn_base_channels,
        ).to(self.device)
        self.depth_predictor_optimizer = optim.Adam(
            self.depth_predictor.parameters(),
            lr=self.depth_predictor_cfg.lr,
            weight_decay=self.depth_predictor_cfg.weight_decay,
        )
        depth_noise = getattr(self.env, "depth_noise", None)
        if depth_noise is None:
            raise ValueError("DawnRunner requires the runtime env to expose depth_noise.")
        live_depth_noise = self._build_adapter_depth_noise(depth_noise)

        sequence_depth_noise = self._build_adapter_depth_noise(depth_noise)
        self._depth_adapter = DAWNDepthAdapter(
            depth_predictor=self.depth_predictor,
            depth_noise=live_depth_noise,
            sequence_depth_noise=sequence_depth_noise,
            depth_index=self._depth_index,
            device=self.device,
        )

        self.actor_critic = ActorCritic(
            obs_dict=obs_dict,
            num_actions=self.env.num_actions,
            use_critic=True,
            policy_cfg=self.policy_cfg,
        ).to(self.device)
        print(self.actor_critic)
        obs_shape_dict = {key: obs_dict[key].shape[1:] for key in self.actor_critic.obs_keys}

        amp_data = AMPLoader(
            device=self.device,
            time_between_frames=self.env.unwrapped.cfg.sim.dt,
            preload_transitions=True,
            num_preload_transitions=self.amp_cfg.amp_num_preload_transitions,
            motion_files=self.amp_cfg.amp_motion_files,
        )
        amp_normalizer = Normalizer(amp_data.observation_dim)

        discriminator = AMPDiscriminator(
            amp_data.observation_dim * 2,
            self.amp_cfg.amp_reward_coef,
            self.amp_cfg.amp_hidden_layer_sizes,
            self.device,
            self.amp_cfg.amp_task_reward_lerp,
        ).to(self.device)

        min_normalized_std = torch.tensor(self.amp_cfg.min_normalized_std, device=self.device)
        dof_range = torch.abs(self.env.dof_pos_limits[0, :, 1] - self.env.dof_pos_limits[0, :, 0])
        min_std = min_normalized_std * dof_range
        self.alg = PPOAMPAlgorithm(
            actor_critic=self.actor_critic,
            discriminator=discriminator,
            amp_data=amp_data,
            amp_normalizer=amp_normalizer,
            device=self.device,
            min_std=min_std,
            algorithm_cfg=self.alg_cfg,
        )
        self.num_steps_per_env = self.runner_cfg.num_steps_per_env
        self.save_interval = self.runner_cfg.save_interval
        self.save_count = 0

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            obs_shape_dict,
            (self.env.num_actions,),
        )

        self.log_dir = log_dir
        self.logging_manager = LoggingManager(log_dir=log_dir, device=self.device)

        self.logging_cfg = logging_cfg

        self.logging_manager.register_plugin(PolicyLogger(device=self.device))
        self.logging_manager.register_plugin(WorldModelLogger(device=self.device))
        self.logging_manager.register_plugin(DepthPredictorLogger(device=self.device))

        self.logging_manager.initialize_env_tracking(self.env.num_envs)

        self.current_learning_iteration = 0
        self.git_status_repos = [__file__]

        _, _ = self.env.reset()

    @property
    def world_model(self) -> DAWNWorldModel:
        return self._world_model

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool) -> None:
        self.logging_manager.initialize_writer(
            logging_cfg=self.logging_cfg,
            env_cfg_dict=self.env.cfg.to_dict(),
            train_cfg_dict=self.cfg.to_dict(),
        )

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs_dict, _ = self.env.get_observations()
        amp_obs = obs_dict["amp_obs"]

        self.train_mode()
        self.alg.discriminator.train()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        sum_wm_storage_size = 0
        wm_is_first = torch.ones(self.env.num_envs, device=self.device)

        wm_obs = self._build_wm_obs(obs_dict, wm_is_first)

        wm_action_history = torch.zeros(
            size=(self.env.num_envs, self.wm_update_interval, self.env.num_actions),
            device=self.device,
        )
        rollout_state = DawnRolloutState(
            wm_obs=wm_obs,
            wm_is_first=wm_is_first,
            wm_action_history=wm_action_history,
        )
        zero_feature = torch.zeros((self.env.num_envs, self.wm_feature_dim), device=self.device)

        self.wm_storage = DAWNRolloutStorage(
            num_envs=self.env.num_envs,
            num_actions=self.env.num_actions,
            max_episode_length=self.env.max_episode_length,
            update_interval=self.wm_update_interval,
            prop_dim=self._obs_spec.prop_with_command_dim,
            depth_shape=self._obs_spec.depth_shape,
            forward_height_map_dim=self._obs_spec.forward_height_map_dim,
            depth_index=self._depth_index,
            camera_buffer_index_by_env=self._camera_buffer_index_by_env,
        )

        for it in range(start_iter, tot_iter):
            self.env.iteration = it
            start = time.time()

            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    if rollout_state.global_step % self.wm_update_interval == 0:
                        wm_embed = self._world_model.rssm.obs_embedder(rollout_state.wm_obs)
                        wm_latent = self._world_model.rssm.observe_step(
                            rollout_state.wm_latent,
                            rollout_state.wm_action,
                            wm_embed,
                            rollout_state.wm_obs["is_first"],
                        )
                        rollout_state.mark_world_model_updated(wm_latent)

                    wm_feature = rollout_state.wm_latent.h if rollout_state.wm_latent is not None else zero_feature
                    obs_dict["wm_feature"] = wm_feature
                    actions = self.alg.act(obs_dict)
                    obs_dict, rewards, dones, infos = self.env.step(actions)
                    depth_noisy = obs_dict.get("depth")
                    depth_clean = obs_dict.get("depth_clean")
                    obs_dict["wm_feature"] = wm_feature

                    next_amp_obs = obs_dict["amp_obs"]
                    rollout_state.advance_step(actions)
                    rollout_state.wm_obs = self._build_wm_obs(obs_dict, rollout_state.wm_is_first)

                    reset_env_ids = infos["reset_env_ids"].cpu().numpy()

                    if len(reset_env_ids) > 0:
                        sum_wm_storage_size = self.wm_storage.complete_episodes(reset_env_ids)
                        rollout_state.reset_finished(reset_env_ids)
                        self._depth_adapter.reset_live_noise(reset_env_ids)

                    if rollout_state.global_step % self.wm_update_interval == 0:
                        self._update_wm_camera_inputs(obs_dict, rollout_state.wm_obs, depth_noisy)
                        self._store_camera_rollout_step(obs_dict, depth_clean, depth_noisy)

                        not_reset_env_ids = rollout_state.active_env_ids()
                        if len(not_reset_env_ids) > 0:
                            wm_action = rollout_state.wm_action
                            if wm_action is None:
                                raise RuntimeError("DAWN rollout action history is not initialized.")
                            self.wm_storage.store_common_step(
                                not_reset_env_ids,
                                rollout_state.wm_obs,
                                wm_action,
                            )

                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    if len(reset_env_ids) > 0:
                        next_amp_obs_with_term[reset_env_ids] = infos["pre_reset_obs"]["amp_obs"][reset_env_ids]

                    rewards = self.alg.discriminator.predict_amp_reward(
                        amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer
                    )[0]
                    amp_obs = torch.clone(next_amp_obs)

                    self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)
                    self.logging_manager.collect_step_metrics(actions, rewards, dones, infos)
                    self.logging_manager.collect_episode_metrics(dones)

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(obs_dict)

            algorithm_metrics = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            algorithm_metrics["mean_noise_std"] = self.actor_critic.action_std.mean().item()

            start_time = time.time()
            if sum_wm_storage_size > self.wm_cfg.algorithm.train_start_steps:
                if it % self.depth_predictor_cfg.train_interval == 0:
                    algorithm_metrics["depth_predictor_loss"] = self.train_depth_predictor()

                wm_metrics = self.train_world_model()
                for name, value in wm_metrics.items():
                    if isinstance(value, np.ndarray | torch.Tensor):
                        value = _mean_scalar(value)
                    algorithm_metrics[name] = value

            world_model_training_time = time.time() - start_time
            algorithm_metrics["extra_time"] = world_model_training_time
            algorithm_metrics["extra_time_name"] = "world model"

            self.logging_manager.log(
                iteration=it,
                total_iterations=tot_iter,
                collection_time=collection_time,
                learn_time=learn_time,
                num_steps_per_env=self.num_steps_per_env,
                num_envs=self.env.num_envs,
                algorithm_metrics=algorithm_metrics,
            )

            if it % (((self.save_count + 1) ** 2) * self.save_interval) == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}"))
                self.save_count += 1

            if it == start_iter:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if git_file_paths:
                    for path in git_file_paths:
                        self.logging_manager.save(path)

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}"), push_to_wandb=True)
        self.logging_manager.finish()

    def _build_adapter_depth_noise(self, runtime_depth_noise: DepthNoise) -> DepthNoise:
        return DepthNoise(near_clip=runtime_depth_noise.near_clip, far_clip=runtime_depth_noise.far_clip)

    def _build_wm_obs(
        self,
        obs_dict: dict[str, torch.Tensor],
        wm_is_first: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        validate_world_model_observation_keys(obs_dict, ("prop_hist", "prop_clean", "command"))
        command = obs_dict["command"]
        wm_obs = build_world_model_obs_template(
            prop=obs_dict["prop_hist"][:, -1],
            command=command,
            wm_is_first=wm_is_first,
            device=self.device,
            depth_shape=self._obs_spec.depth_shape,
        )
        wm_obs["prop_clean"] = torch.cat([obs_dict["prop_clean"], command], dim=-1).to(self.device)
        return wm_obs

    def _store_camera_rollout_step(
        self,
        obs_dict: dict[str, torch.Tensor],
        depth_clean: torch.Tensor | None,
        depth_noisy: torch.Tensor | None,
    ) -> None:
        if depth_clean is None or depth_noisy is None:
            raise ValueError("DAWN camera runtime requires both clean and noisy depth observations.")
        self.wm_storage.store_real_depth_step(obs_dict, depth_clean, depth_noisy)

    def _update_wm_camera_inputs(
        self,
        obs_dict: dict[str, torch.Tensor],
        wm_obs: dict[str, torch.Tensor],
        depth_noisy: torch.Tensor | None,
    ) -> None:
        wm_obs["image"] = self._depth_adapter.build_live_depth_input(
            forward_heightmap=obs_dict["forward_height_map"],
            prop_clean=wm_obs["prop_clean"],
            depth_noisy=depth_noisy,
        )

    def _sample_depth_predictor_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.wm_storage.sample_depth_predictor_batch(
            batch_size=self.depth_predictor_cfg.batch_size,
            device=self.device,
        )

    def _collect_world_model_image_batch(
        self,
        batch_idx: np.ndarray,
        batch_end_idx: list[int],
        batch_length: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        synth_noisy, synth_clean = self._predict_world_model_depth_sequences(batch_idx, batch_end_idx, batch_length)

        depth_noisy_values: list[torch.Tensor] = []
        depth_clean_values: list[torch.Tensor] = []
        for pos, (idx, end_idx) in enumerate(zip(batch_idx, batch_end_idx)):
            real_depth_sequence = self.wm_storage.real_depth_sequence(
                env_idx=int(idx),
                end_idx=end_idx,
                batch_length=batch_length,
            )
            if real_depth_sequence is None:
                depth_noisy_values.append(synth_noisy[pos])
                depth_clean_values.append(synth_clean[pos])
                continue
            depth_noisy, depth_clean = real_depth_sequence
            depth_noisy_values.append(depth_noisy)
            depth_clean_values.append(depth_clean)
        return depth_noisy_values, depth_clean_values

    def _predict_world_model_depth_sequences(
        self,
        batch_idx: np.ndarray,
        batch_end_idx: list[int],
        batch_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        forward_heightmap_list: list[torch.Tensor] = []
        prop_list: list[torch.Tensor] = []
        for idx, end_idx in zip(batch_idx, batch_end_idx):
            forward_heightmap, prop = self.wm_storage.prediction_inputs(
                env_idx=int(idx),
                end_idx=end_idx,
                batch_length=batch_length,
                device=self.device,
            )
            forward_heightmap_list.append(forward_heightmap)
            prop_list.append(prop)
        return self._depth_adapter.predict_depth_sequences(
            forward_heightmap=torch.stack(forward_heightmap_list),
            prop_clean=torch.stack(prop_list),
        )

    def train_depth_predictor(self) -> float:
        total_mse_loss = 0.0
        for _ in range(self.depth_predictor_cfg.training_iters):
            forward_heightmap, prop, depth_clean = self._sample_depth_predictor_batch()

            predicted_depth_clean = self.depth_predictor(forward_heightmap, prop)
            depth_predict_loss = (depth_clean - predicted_depth_clean).pow(2).mean()
            depth_predict_loss *= self.depth_predictor_cfg.loss_scale
            self.depth_predictor_optimizer.zero_grad()
            depth_predict_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.depth_predictor.parameters(), 1)
            self.depth_predictor_optimizer.step()
            total_mse_loss += depth_predict_loss.detach().item() / self.depth_predictor_cfg.loss_scale
        return total_mse_loss / self.depth_predictor_cfg.training_iters

    def train_world_model(self) -> dict[str, float]:
        wm_metrics = {}
        mets = {}
        for i in range(self.wm_cfg.algorithm.train_steps_per_iter):
            sampled_indices = self.wm_storage.sample_world_model_indices(
                batch_size=self.wm_cfg.algorithm.batch_size,
                max_batch_length=self.wm_cfg.algorithm.batch_length,
            )
            if sampled_indices is None:
                continue
            batch_idx, batch_length, batch_end_idx = sampled_indices

            batch_data = self.wm_storage.collect_world_model_common_batch(
                batch_idx=batch_idx,
                batch_end_idx=batch_end_idx,
                batch_length=batch_length,
            )

            depth_noisy_values, depth_clean_values = self._collect_world_model_image_batch(
                batch_idx=batch_idx,
                batch_end_idx=batch_end_idx,
                batch_length=batch_length,
            )
            batch_data.update(
                {
                    "image": torch.stack(depth_noisy_values),
                    "image_clean": torch.stack(depth_clean_values),
                }
            )

            is_first = torch.zeros((self.wm_cfg.algorithm.batch_size, batch_length))
            is_first[:, 0] = 1
            batch_data["is_first"] = is_first
            _post, _context, mets = self._world_model._train(batch_data)  # noqa: SLF001

        wm_metrics.update(mets)
        return wm_metrics

    def save(self, model_dir: str, infos: dict | None = None, push_to_wandb: bool = False) -> None:
        os.makedirs(model_dir, exist_ok=True)
        self.actor_critic.save(model_dir)
        self._world_model.save(model_dir)
        self.depth_predictor.save(model_dir)

        if push_to_wandb:
            self.logging_manager.save(model_dir)

    def get_inference_policy(self, device: str | None = None) -> Callable[[torch.Tensor], torch.Tensor]:
        self.eval_mode()
        if device is not None:
            self.actor_critic.to(device)
        return self.actor_critic.act_inference

    def train_mode(self) -> None:
        self.actor_critic.train_mode()

    def eval_mode(self) -> None:
        self.actor_critic.eval_mode()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.git_status_repos.append(repo_file_path)
