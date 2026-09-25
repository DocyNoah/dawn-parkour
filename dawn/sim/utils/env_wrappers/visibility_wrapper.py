from __future__ import annotations

from typing import TYPE_CHECKING

import omni.usd  # type: ignore
from isaacsim.core.utils.prims import get_prim_at_path  # type: ignore
from pxr import Usd, UsdGeom  # type: ignore

from dawn.sim.utils.env_wrappers.vec_env_wrapper import VecEnvWrapper

if TYPE_CHECKING:
    import torch

    from dawn.sim.envs.manager_based_rl_amp_env import ManagerBasedRLAMPEnv


class VisibilityWrapper(VecEnvWrapper):
    def __init__(self, env: ManagerBasedRLAMPEnv | VecEnvWrapper):
        super().__init__(env)
        self.log_visibility = True

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs_dict, extras = self.env.reset()
        if self.unwrapped.cfg.sim.enable_cameras and self.unwrapped.cfg.invisible_robot:
            for i in range(self.num_envs):
                robot_path = f"/World/envs/env_{i}/Robot"
                self._force_set_visibility(robot_path, "invisible", self.log_visibility)
                camera_path = f"{robot_path}/trunk/custom_cam"
                camera_prim = get_prim_at_path(camera_path)
                if camera_prim.IsValid():
                    self._force_set_visibility(camera_path, "invisible", self.log_visibility)
                    self.log_visibility = False
        return obs_dict, extras

    def _force_set_visibility(self, prim_path: str, visibility: str = "invisible", print_info: bool = True) -> None:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            if print_info:
                print(f"[WARNING] Invalid prim: {prim_path}")
            return
        imageable = UsdGeom.Imageable(prim)
        imageable.CreateVisibilityAttr().Set(
            UsdGeom.Tokens.invisible if visibility == "invisible" else UsdGeom.Tokens.visible
        )
        if print_info:
            v = imageable.ComputeVisibility(Usd.TimeCode.Default())
            print(f"[[FORCED] {prim_path}] visibility = {v}")
