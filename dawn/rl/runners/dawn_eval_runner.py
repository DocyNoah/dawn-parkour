from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from dawn.rl.configs.train_cfg import DawnTrainCfg
from dawn.rl.modules import DAWNWorldModel, DepthPredictor
from dawn.rl.runners.eval_runner import EvalRunner
from dawn.rl.runners.utils import RewardTracker, SimulationApp
from dawn.rl.runners.utils.world_model_contracts import (
    build_world_model_obs_template,
    infer_depth_shape,
    infer_world_model_depth_index,
    infer_world_model_observation_spec,
    validate_world_model_observation_keys,
)

if TYPE_CHECKING:
    from dawn.rl.networks.dreamer.rssm import Latent


class DawnEvalRunner(EvalRunner):
    @property
    def world_model(self) -> DAWNWorldModel:
        return self._world_model

    def _prepare_actor_obs(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        infer_world_model_depth_index(self.env)
        obs_spec = infer_world_model_observation_spec(obs_dict)
        if not isinstance(self.cfg, DawnTrainCfg):
            raise TypeError("DawnEvalRunner requires DawnTrainCfg.")
        self.wm_cfg = self.cfg.wm
        wm_update_interval = int(self.env.cfg.depth.update_interval)
        wm_obs_shape_dict = {
            "prop": (obs_spec.prop_with_command_dim,),
            "image": obs_spec.depth_shape,
        }
        print("Building world model...")
        self._world_model = DAWNWorldModel(
            config=self.wm_cfg,
            obs_shape=wm_obs_shape_dict,
            device=self.device,
            num_actions=self.env.num_actions * wm_update_interval,
        ).to(self.device)
        print("Finished building world model.")
        self.wm_feature_dim = self.wm_cfg.rssm.dyn_deter
        self.depth_predictor = DepthPredictor(
            forward_heightmap_dim=obs_spec.forward_height_map_dim,
            prop_dim=obs_spec.prop_with_command_dim,
            depth_image_dims=obs_spec.depth_shape[-2:],
            cnn_base_channels=self.cfg.depth_predictor.cnn_base_channels,
        ).to(self.device)

        obs_dict["wm_feature"] = torch.zeros(self.env.num_envs, self.wm_feature_dim, device=self.device)
        return obs_dict

    def load(self, model_dir: str) -> None:
        super().load(model_dir)
        print("[INFO]: ========== Start loading world model ==========")
        self._world_model.load(model_dir)
        self.depth_predictor.load(model_dir)
        print("[INFO]: ========== Done loading world model ==========")

    def _update_wm(
        self,
        timestep: int,
        wm_update_interval: int,
        obs_dict: dict[str, torch.Tensor],
        wm_obs: dict[str, torch.Tensor],
        wm_latent: Latent | None,
        wm_action: torch.Tensor | None,
        wm_is_first: torch.Tensor,
    ) -> tuple[Latent | None, torch.Tensor]:
        if timestep % wm_update_interval != 0:
            return wm_latent, obs_dict["wm_feature"]

        wm_obs["image"] = obs_dict["depth"].to(self.env.device)
        wm_embed = self.world_model.rssm.obs_embedder(wm_obs)
        wm_latent = self.world_model.rssm.observe_step(wm_latent, wm_action, wm_embed, wm_obs["is_first"])
        wm_is_first[:] = 0
        obs_dict["wm_feature"] = wm_latent.h
        return wm_latent, wm_latent.h

    def _advance_wm_state(
        self,
        obs_dict: dict[str, torch.Tensor],
        infos: dict[str, object],
        actions: torch.Tensor,
        wm_feature: torch.Tensor,
        wm_action_history: torch.Tensor,
        wm_is_first: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        obs_dict["wm_feature"] = wm_feature

        wm_action_history = torch.cat([wm_action_history[:, 1:], actions.unsqueeze(1)], dim=1)
        wm_obs = build_world_model_obs_template(
            prop=obs_dict["prop_hist"][:, -1],
            command=obs_dict["command"],
            wm_is_first=wm_is_first,
            device=self.env.device,
            depth_shape=infer_depth_shape(obs_dict["depth"]),
        )

        reset_env_ids_tensor = infos["reset_env_ids"]
        if not isinstance(reset_env_ids_tensor, torch.Tensor):
            raise TypeError("infos['reset_env_ids'] must be a torch.Tensor.")
        reset_env_ids = reset_env_ids_tensor.cpu().numpy()
        if len(reset_env_ids) > 0:
            wm_action_history[reset_env_ids, :] = 0
            wm_is_first[reset_env_ids] = 1

        wm_action = wm_action_history.flatten(1)
        return wm_obs, wm_action_history, wm_action

    def run(self, simulation_app: SimulationApp, real_time: bool, video_length: int | None) -> None:
        dt = self.env.unwrapped.physics_dt
        timestep = 0
        tracker = RewardTracker(env=self.env)
        obs_dict, _ = self.env.get_observations()

        validate_world_model_observation_keys(obs_dict, ("prop_hist", "command", "depth"))
        self.world_model.to(self.env.device)
        wm_feature = torch.zeros((self.env.num_envs, self.wm_feature_dim), device=self.env.device)
        obs_dict["wm_feature"] = wm_feature

        wm_obs = build_world_model_obs_template(
            prop=obs_dict["prop_hist"][:, -1],
            command=obs_dict["command"],
            wm_is_first=torch.ones(self.env.num_envs, device=self.env.device),
            device=self.env.device,
            depth_shape=infer_depth_shape(obs_dict["depth"]),
        )
        wm_update_interval = int(self.env.cfg.depth.update_interval)
        wm_action_history = torch.zeros(
            size=(self.env.num_envs, wm_update_interval, self.env.num_actions),
            device=self.env.device,
        )
        wm_latent: Latent | None = None
        wm_action: torch.Tensor | None = None
        wm_is_first = wm_obs["is_first"]

        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                wm_latent, wm_feature = self._update_wm(
                    timestep=timestep,
                    wm_update_interval=wm_update_interval,
                    obs_dict=obs_dict,
                    wm_obs=wm_obs,
                    wm_latent=wm_latent,
                    wm_action=wm_action,
                    wm_is_first=wm_is_first,
                )

                actions = self.actor.act_inference(obs_dict)
                obs_dict, reward, dones, infos = self.env.step(actions)
                self.actor.reset(dones)

                wm_obs, wm_action_history, wm_action = self._advance_wm_state(
                    obs_dict=obs_dict,
                    infos=infos,
                    actions=actions,
                    wm_feature=wm_feature,
                    wm_action_history=wm_action_history,
                    wm_is_first=wm_is_first,
                )

            tracker.update(reward, dones)
            tracker.maybe_print()

            timestep += 1
            if video_length is not None and timestep == video_length:
                break

            sleep_time = dt - (time.time() - start_time)
            if real_time and sleep_time > 0:
                time.sleep(sleep_time)
