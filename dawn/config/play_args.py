from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field, fields
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

import tyro

from dawn.config.experiment_cfg import AppCfg, EnvCfg, EnvWrapperCfg, VideoCfg

if TYPE_CHECKING:
    from dawn.config.experiment_cfg import ExperimentCfg

T = TypeVar("T")
OptionalFlag = tyro.conf.DisallowNone[bool | None]


class LoggedOverrideCfg(Generic[T]):
    def apply_to(self, base_cfg: T) -> T:
        loaded_cfg = copy(base_cfg)
        for override_field in fields(type(self)):
            value = getattr(self, override_field.name)
            if value is not None:
                setattr(loaded_cfg, override_field.name, value)
        return loaded_cfg


@dataclass
class CheckpointCfg:
    model_dir: Path


@dataclass
class PlayEnvCfg(LoggedOverrideCfg[EnvCfg]):
    task: str | None = "Go1-DAWN-Play-v0"
    seed: int | None = None
    num_envs: int | None = None
    depth_image_shape: tuple[int, int] | None = None
    invisible_robot: OptionalFlag = None


@dataclass
class PlayAppCfg(LoggedOverrideCfg[AppCfg]):
    headless: OptionalFlag = None
    enable_cameras: OptionalFlag = None
    device: str | None = None
    livestream: Literal[0, 1, 2] | None = None
    xr: OptionalFlag = None
    verbose: OptionalFlag = None
    info: OptionalFlag = None
    experience: str | None = None
    rendering_mode: Literal["performance", "balanced", "quality"] | None = None
    kit_args: str | None = None


@dataclass
class PlayVideoCfg(LoggedOverrideCfg[VideoCfg]):
    enable: OptionalFlag = None
    length: int | None = None
    interval: int | None = None


@dataclass
class PlayWrapperCfg(LoggedOverrideCfg[EnvWrapperCfg]):
    use_observation_wrapper: OptionalFlag = None
    use_depth_buffer_wrapper: OptionalFlag = None
    use_depth_noise_wrapper: OptionalFlag = None
    use_visibility_wrapper: OptionalFlag = None
    use_camera_follow_wrapper: OptionalFlag = None
    use_visualization_wrapper: OptionalFlag = None


@dataclass
class PlaybackCfg:
    disable_fabric: bool = False
    real_time: bool = True
    export: bool = False


@dataclass
class PlayCfg:
    checkpoint: CheckpointCfg
    app: PlayAppCfg = field(default_factory=PlayAppCfg)
    env: PlayEnvCfg = field(default_factory=PlayEnvCfg)
    video: PlayVideoCfg = field(default_factory=PlayVideoCfg)
    wrappers: PlayWrapperCfg = field(default_factory=PlayWrapperCfg)
    playback: PlaybackCfg = field(default_factory=PlaybackCfg)


@dataclass
class LoadedPlayCfg:
    checkpoint: CheckpointCfg
    app: AppCfg
    env: EnvCfg
    video: VideoCfg
    wrappers: EnvWrapperCfg
    playback: PlaybackCfg


def load_play_cfg(cfg: PlayCfg, experiment_cfg: ExperimentCfg) -> LoadedPlayCfg:
    env_cfg = cfg.env.apply_to(experiment_cfg.env)
    env_cfg.camera_num_envs = env_cfg.num_envs
    return LoadedPlayCfg(
        checkpoint=cfg.checkpoint,
        app=cfg.app.apply_to(experiment_cfg.app),
        env=env_cfg,
        video=cfg.video.apply_to(experiment_cfg.video),
        wrappers=cfg.wrappers.apply_to(experiment_cfg.wrappers),
        playback=cfg.playback,
    )


__all__ = [
    "CheckpointCfg",
    "LoadedPlayCfg",
    "PlayAppCfg",
    "PlayCfg",
    "PlayEnvCfg",
    "PlayVideoCfg",
    "PlayWrapperCfg",
    "PlaybackCfg",
    "load_play_cfg",
]
