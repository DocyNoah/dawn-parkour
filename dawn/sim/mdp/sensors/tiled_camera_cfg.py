# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.utils import configclass

from dawn.sim.mdp.sensors.tiled_camera import TiledCamera


@configclass
class TiledCameraCfg(CameraCfg):
    class_type: type = TiledCamera
