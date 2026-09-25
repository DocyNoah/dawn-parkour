from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
    from torch import nn

    from dawn.utils.env_builder import VecEnvWrapper


def export_policy(
    enabled: bool,
    actor: nn.Module,
    env: VecEnvWrapper,
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    model_dir: str,
) -> None:
    if not enabled:
        return

    from dawn.utils.exporter import export_policy_as_jit

    export_model_dir = os.path.join(os.path.dirname(model_dir), "exported")
    export_policy_as_jit(
        actor=actor,
        path=export_model_dir,
        filename="policy_cpu.pt",
        device="cpu",
        env_cfg=env_cfg,
    )
    export_policy_as_jit(
        actor=actor,
        path=export_model_dir,
        filename=f"policy_{env.unwrapped.device}.pt",
        device=env.unwrapped.device,
        env_cfg=env_cfg,
    )


def validate_model_dir(model_dir: str) -> None:
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory: {model_dir} does not exist.")
