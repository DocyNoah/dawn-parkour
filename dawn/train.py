from __future__ import annotations

import random
import sys
import traceback
from datetime import datetime
from typing import TYPE_CHECKING, Any

import torch
import tyro

from dawn.config.train_presets import get_train_presets
from dawn.rl.configs.factory import runner_cls_from_algorithm_name
from dawn.utils.env_builder import build_env, validate_env_args
from dawn.utils.simulation import load_task_env_cfg, register_tasks

if TYPE_CHECKING:
    from dawn.config.experiment_cfg import ExperimentCfg, TorchBackendCfg


def set_train_seed(cfg: ExperimentCfg) -> ExperimentCfg:
    if cfg.env.seed == -1:
        cfg.env.seed = random.randint(0, 10000)
    return cfg


def fill_logging_timestamp(cfg: ExperimentCfg) -> ExperimentCfg:
    if not cfg.logging.now_time:
        cfg.logging.now_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return cfg


def _configure_torch_backend(backend_cfg: TorchBackendCfg) -> None:
    torch.backends.cuda.matmul.allow_tf32 = backend_cfg.cuda_matmul_allow_tf32
    torch.backends.cudnn.allow_tf32 = backend_cfg.cudnn_allow_tf32
    torch.backends.cudnn.deterministic = backend_cfg.cudnn_deterministic
    torch.backends.cudnn.benchmark = backend_cfg.cudnn_benchmark


def _close_after_train(resource: Any, name: str, suppress_cleanup_error: bool) -> BaseException | None:
    try:
        resource.close()
    except Exception as exc:
        if suppress_cleanup_error:
            print(f"[ERROR] Failed to close {name}; preserving the training traceback.", file=sys.stderr)
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
            return None
        return exc
    return None


def main(cfg: ExperimentCfg) -> None:
    cfg = set_train_seed(cfg)
    cfg = fill_logging_timestamp(cfg)
    if cfg.video.enable and not cfg.app.enable_cameras:
        print("[INFO] video.enable is set; enabling app.enable_cameras for this run.")
        cfg.app.enable_cameras = True
    validate_env_args(cfg.env.num_envs, cfg.env.camera_num_envs, cfg.env.depth_image_shape)
    if cfg.app.enable_cameras and not cfg.env.invisible_robot:
        print(
            "[WARN] Cameras are enabled while invisible_robot is disabled. "
            "Robots may appear in each other's camera frames and hurt training quality."
        )
    _configure_torch_backend(cfg.torch_backend)

    from dawn.utils.launch import launch_app
    from dawn.utils.log_setup import get_isaaclab_version_path, setup_log

    simulation_app = launch_app(cfg.app)

    dawn_envs = register_tasks()

    env_cfg = load_task_env_cfg(
        cfg.env.task,
        cfg.app,
        cfg.env,
    )

    log_dir = setup_log(env_cfg=env_cfg, experiment_cfg=cfg, sys_argv=sys.argv)
    env = build_env(
        cfg.env.task,
        env_cfg,
        headless=cfg.app.headless,
        video=cfg.video,
        video_mode="train",
        log_dir=str(log_dir),
        wrappers=cfg.wrappers,
    )

    training_failed = False
    try:
        train_cfg = cfg.train
        runner_cls = runner_cls_from_algorithm_name(train_cfg.algorithm_name)
        runner = runner_cls(
            env,
            train_cfg,
            log_dir=str(log_dir),
            device=cfg.app.device,
            logging_cfg=cfg.logging,
        )
        runner.add_git_repo_to_log(__file__)
        runner.add_git_repo_to_log(dawn_envs.__file__)
        runner.add_git_repo_to_log(get_isaaclab_version_path())

        runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    except BaseException as exc:
        training_failed = True
        print("[ERROR] Training failed with exception:", file=sys.stderr)
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        raise
    finally:
        cleanup_error = _close_after_train(env, "environment", training_failed)
        app_cleanup_error = _close_after_train(simulation_app, "simulation app", training_failed)
        if cleanup_error is not None:
            raise cleanup_error
        if app_cleanup_error is not None:
            raise app_cleanup_error


if __name__ == "__main__":
    main(tyro.extras.overridable_config_cli(get_train_presets()))
