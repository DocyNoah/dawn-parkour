from typing import TYPE_CHECKING, Any

from dawn.config.experiment_cfg import ExperimentCfg
from dawn.config.train_presets import get_train_presets

if TYPE_CHECKING:
    from dawn.config.loader import load_logged_experiment_cfg
    from dawn.config.play_args import (
        CheckpointCfg,
        LoadedPlayCfg,
        PlayAppCfg,
        PlaybackCfg,
        PlayCfg,
        PlayEnvCfg,
        PlayVideoCfg,
        PlayWrapperCfg,
        load_play_cfg,
    )


def __getattr__(name: str) -> Any:
    if name == "load_logged_experiment_cfg":
        from dawn.config.loader import load_logged_experiment_cfg as _load_logged_experiment_cfg

        return _load_logged_experiment_cfg
    if name in {
        "CheckpointCfg",
        "LoadedPlayCfg",
        "PlayAppCfg",
        "PlayCfg",
        "PlayEnvCfg",
        "PlayVideoCfg",
        "PlayWrapperCfg",
        "PlaybackCfg",
        "load_play_cfg",
    }:
        from dawn.config import play_args as _play_args

        return getattr(_play_args, name)
    raise AttributeError(name)


__all__ = [
    "CheckpointCfg",
    "ExperimentCfg",
    "LoadedPlayCfg",
    "PlayAppCfg",
    "PlayCfg",
    "PlayEnvCfg",
    "PlayVideoCfg",
    "PlayWrapperCfg",
    "PlaybackCfg",
    "get_train_presets",
    "load_logged_experiment_cfg",
    "load_play_cfg",
]
