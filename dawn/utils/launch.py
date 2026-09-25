from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from isaaclab.app import AppLauncher

if TYPE_CHECKING:
    from isaacsim.simulation_app import SimulationApp

    from dawn.config.experiment_cfg import AppCfg


_APP_LAUNCHER: AppLauncher | None = None


def launch_app(app_cfg: AppCfg) -> SimulationApp:
    global _APP_LAUNCHER

    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        _APP_LAUNCHER = AppLauncher(app_cfg.to_app_launcher_dict())
    finally:
        sys.argv = original_argv

    return _APP_LAUNCHER.app


__all__ = ["launch_app"]
