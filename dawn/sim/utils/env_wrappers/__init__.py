from __future__ import annotations

from importlib import import_module
from typing import Any

_MODULE_BY_EXPORT = {
    "CameraFollowWrapper": ".camera_follow_wrapper",
    "ChannelFirstWrapper": ".channel_first_wrapper",
    "DepthBufferWrapper": ".depth_buffer_wrapper",
    "DepthNoiseWrapper": ".depth_noise_wrapper",
    "EnvIndexWrapper": ".env_index_wrapper",
    "IsaaclabWrapper": ".isaaclab_wrapper",
    "ObservationWrapper": ".observation_wrapper",
    "VecEnvWrapper": ".vec_env_wrapper",
    "VisibilityWrapper": ".visibility_wrapper",
    "VisualizationWrapper": ".visualization_wrapper",
}

__all__ = tuple(_MODULE_BY_EXPORT)


def __getattr__(name: str) -> Any:
    if name not in _MODULE_BY_EXPORT:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_MODULE_BY_EXPORT[name], __name__)
    return getattr(module, name)
