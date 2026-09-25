from __future__ import annotations

from dawn.rl.configs.train_cfg import DawnTrainCfg
from dawn.rl.configs.world_model_cfg import DAWNWorldModelCfg


def runner_cls_from_algorithm_name(algorithm_name: str) -> type:
    if algorithm_name != "dawn":
        raise ValueError(f"Unsupported algorithm name: {algorithm_name}")
    from dawn.rl.runners.dawn_runner import DawnRunner

    return DawnRunner


def train_cfg_from_algorithm_name(algorithm_name: str) -> DawnTrainCfg:
    if algorithm_name != "dawn":
        raise ValueError(f"Unsupported algorithm name: {algorithm_name}")
    return DawnTrainCfg(algorithm_name="dawn", wm=DAWNWorldModelCfg())
