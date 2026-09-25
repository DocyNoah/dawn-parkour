# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.envs.manager_based_env_cfg import ManagerBasedEnvCfg
from isaaclab.envs.ui import ManagerBasedRLEnvWindow
from isaaclab.utils import configclass


@configclass
class ManagerBasedRLAMPEnvCfg(ManagerBasedEnvCfg):
    ui_window_class_type: type | None = ManagerBasedRLEnvWindow

    is_finite_horizon: bool = False

    episode_length_s: float = MISSING

    rewards: object = MISSING

    terminations: object = MISSING

    curriculum: object | None = None

    commands: object | None = None
