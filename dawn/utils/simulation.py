from __future__ import annotations

from typing import TYPE_CHECKING

from dawn.utils.env_builder import update_env_cfg, validate_env_args

if TYPE_CHECKING:
    from types import ModuleType

    from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg

    from dawn.config.experiment_cfg import AppCfg, EnvCfg


def register_tasks() -> ModuleType:
    import gymnasium as gym
    import isaaclab_tasks

    import dawn.sim.envs as dawn_envs

    gym.register_envs(dawn_envs)
    gym.register_envs(isaaclab_tasks)
    return dawn_envs


def load_task_env_cfg(
    task: str | None,
    app_cfg: AppCfg,
    env_cfg: EnvCfg,
    *,
    disable_fabric: bool = False,
) -> ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg:
    from isaaclab_tasks.utils import parse_env_cfg

    if task is None:
        raise ValueError("env.task must be set.")
    validate_env_args(env_cfg.num_envs, env_cfg.camera_num_envs, env_cfg.depth_image_shape)

    if disable_fabric:
        parsed_env_cfg = parse_env_cfg(
            task,
            device=app_cfg.device,
            num_envs=env_cfg.num_envs,
            use_fabric=False,
        )
    else:
        parsed_env_cfg = parse_env_cfg(
            task,
            device=app_cfg.device,
            num_envs=env_cfg.num_envs,
        )

    return update_env_cfg(
        parsed_env_cfg,
        num_envs=env_cfg.num_envs,
        camera_num_envs=env_cfg.camera_num_envs,
        depth_image_shape=env_cfg.depth_image_shape,
        seed=env_cfg.seed,
        device=app_cfg.device,
        invisible_robot=env_cfg.invisible_robot,
        enable_cameras=app_cfg.enable_cameras,
        headless=app_cfg.headless,
    )
