from __future__ import annotations

from os import path
from typing import TYPE_CHECKING

from dawn.config.loader import load_logged_experiment_cfg
from dawn.config.play_args import PlayCfg, load_play_cfg
from dawn.utils.env_builder import build_env
from dawn.utils.model_io import export_policy, validate_model_dir
from dawn.utils.simulation import load_task_env_cfg, register_tasks

if TYPE_CHECKING:
    from dawn.utils.env_builder import VecEnvWrapper


def run_playback(cfg: PlayCfg, runner_cls: type) -> None:
    env: VecEnvWrapper | None = None
    simulation_app = None

    try:
        model_dir = str(cfg.checkpoint.model_dir)
        validate_model_dir(model_dir)

        experiment_cfg = load_logged_experiment_cfg(path.dirname(model_dir))
        cfg = load_play_cfg(cfg, experiment_cfg)
        task = cfg.env.task

        from dawn.utils.launch import launch_app

        simulation_app = launch_app(cfg.app)

        register_tasks()
        env_cfg = load_task_env_cfg(
            task,
            cfg.app,
            cfg.env,
            disable_fabric=cfg.playback.disable_fabric,
        )

        env = build_env(
            task=task,
            env_cfg=env_cfg,
            headless=cfg.app.headless,
            video=cfg.video,
            video_mode="play",
            log_dir=path.dirname(model_dir),
            wrappers=cfg.wrappers,
        )
        train_cfg = experiment_cfg.train

        runner = runner_cls(env, train_cfg, device=cfg.app.device)
        runner.load(model_dir)
        export_policy(
            enabled=cfg.playback.export,
            actor=runner.actor.actor,
            env=env,
            env_cfg=env_cfg,
            model_dir=model_dir,
        )

        runner.run(
            simulation_app=simulation_app,
            real_time=cfg.playback.real_time,
            video_length=cfg.video.length if cfg.video.enable else None,
        )

    except KeyboardInterrupt:
        print("[INFO]: Playback interrupted by user.")
    finally:
        if env is not None:
            env.close()
        if simulation_app is not None:
            simulation_app.close()
