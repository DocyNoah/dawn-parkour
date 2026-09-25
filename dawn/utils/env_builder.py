from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg

    from dawn.config.experiment_cfg import EnvWrapperCfg, VideoCfg
    from dawn.sim.utils.env_wrappers import VecEnvWrapper


DEPTH_WRAPPER_CHAIN: tuple[str, ...] = (
    "IsaaclabWrapper",
    "ChannelFirstWrapper",
    "DepthBufferWrapper",
    "DepthNoiseWrapper",
    "ObservationWrapper",
)


def validate_env_args(
    num_envs: int | None,
    camera_num_envs: int | None,
    depth_image_shape: tuple[int, int] | None = None,
) -> None:
    if num_envs is not None and num_envs <= 0:
        raise ValueError(f"num_envs must be > 0, got {num_envs}")
    if camera_num_envs is not None and camera_num_envs <= 0:
        raise ValueError(f"camera_num_envs must be > 0, got {camera_num_envs}")
    if num_envs is not None and camera_num_envs is not None and camera_num_envs > num_envs:
        raise ValueError(f"camera_num_envs ({camera_num_envs}) must be <= num_envs ({num_envs})")
    if depth_image_shape is not None:
        if len(depth_image_shape) != 2:
            raise ValueError(f"depth_image_shape must be (height, width), got {depth_image_shape}")
        height, width = depth_image_shape
        if height <= 0 or width <= 0:
            raise ValueError(f"depth_image_shape must contain positive values, got {depth_image_shape}")


def env_uses_depth_camera(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> bool:
    observations = getattr(env_cfg, "observations", None)
    if observations is None:
        return False
    return getattr(observations, "depth", None) is not None or getattr(observations, "depth_hist", None) is not None


def disable_non_depth_camera_artifacts(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
) -> None:
    if env_uses_depth_camera(env_cfg):
        return

    if hasattr(env_cfg.scene, "depth_camera"):
        env_cfg.scene.depth_camera = None

    events = getattr(env_cfg, "events", None)
    if events is not None and hasattr(events, "randomize_camera_offset"):
        events.randomize_camera_offset = None


def _apply_depth_camera_overrides(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    camera_num_envs: int | None,
    depth_image_shape: tuple[int, int] | None,
) -> None:
    if not env_uses_depth_camera(env_cfg):
        return

    depth_cfg = getattr(env_cfg, "depth", None)
    if depth_cfg is None:
        return

    if camera_num_envs is not None:
        depth_cfg.camera_num_envs = camera_num_envs

    elif depth_cfg.camera_num_envs is not None:
        pass

    else:
        depth_cfg.camera_num_envs = env_cfg.scene.num_envs

    if depth_image_shape is not None:
        depth_cfg.original_image_shape = depth_image_shape
        depth_cfg.resized_image_shape = depth_image_shape

    if depth_cfg.camera_num_envs > env_cfg.scene.num_envs:
        raise ValueError(
            f"depth.camera_num_envs ({depth_cfg.camera_num_envs}) must be <= scene.num_envs ({env_cfg.scene.num_envs})"
        )

    from dawn.sim.mdp.sensors.sensors_preset import get_depth_camera_cfg

    env_cfg.scene.depth_camera = get_depth_camera_cfg(
        camera_num_envs=depth_cfg.camera_num_envs,
        height=depth_cfg.original_image_shape[0],
        width=depth_cfg.original_image_shape[1],
    )


def update_env_cfg(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    num_envs: int | None,
    camera_num_envs: int | None,
    depth_image_shape: tuple[int, int] | None,
    seed: int,
    device: str,
    invisible_robot: bool,
    enable_cameras: bool,
    headless: bool,
) -> ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg:
    disable_non_depth_camera_artifacts(env_cfg)

    if num_envs is not None:
        env_cfg.scene.num_envs = num_envs

    env_cfg.seed = seed

    env_cfg.sim.device = device
    env_cfg.invisible_robot = invisible_robot
    env_cfg.sim.enable_cameras = enable_cameras
    env_cfg.sim.headless = headless
    _apply_depth_camera_overrides(env_cfg, camera_num_envs, depth_image_shape)
    return env_cfg


def validate_depth_wrapper_setup(has_depth: bool, wrappers: EnvWrapperCfg) -> None:
    if not has_depth:
        return

    depth_noise_enabled = wrappers.use_depth_noise_wrapper
    if depth_noise_enabled and not wrappers.use_depth_buffer_wrapper:
        raise ValueError("Depth noise wrappers require DepthBufferWrapper to preserve delayed-depth semantics.")


def get_runtime_wrapper_order(has_depth: bool, headless: bool, wrappers: EnvWrapperCfg) -> tuple[str, ...]:
    validate_depth_wrapper_setup(has_depth=has_depth, wrappers=wrappers)

    wrapper_order = ["IsaaclabWrapper", "ChannelFirstWrapper"]
    if has_depth and wrappers.use_depth_buffer_wrapper:
        wrapper_order.append("DepthBufferWrapper")
    if has_depth and wrappers.use_depth_noise_wrapper:
        wrapper_order.append("DepthNoiseWrapper")
    if wrappers.use_observation_wrapper:
        wrapper_order.append("ObservationWrapper")
    if wrappers.use_visibility_wrapper:
        wrapper_order.append("VisibilityWrapper")

    camera_follow_enabled = wrappers.use_camera_follow_wrapper or wrappers.use_visualization_wrapper
    if not headless and camera_follow_enabled:
        wrapper_order.append("EnvIndexWrapper")
        wrapper_order.append("CameraFollowWrapper")
    if not headless and wrappers.use_visualization_wrapper:
        wrapper_order.append("VisualizationWrapper")

    if has_depth:
        active_depth_chain = tuple(
            wrapper_name for wrapper_name in wrapper_order if wrapper_name in DEPTH_WRAPPER_CHAIN
        )
        expected_depth_chain = tuple(
            wrapper_name
            for wrapper_name in DEPTH_WRAPPER_CHAIN
            if wrapper_name == "IsaaclabWrapper"
            or wrapper_name == "ChannelFirstWrapper"
            or (wrapper_name == "DepthBufferWrapper" and wrappers.use_depth_buffer_wrapper)
            or (wrapper_name == "DepthNoiseWrapper" and wrappers.use_depth_noise_wrapper)
            or (wrapper_name == "ObservationWrapper" and wrappers.use_observation_wrapper)
        )
        if active_depth_chain != expected_depth_chain:
            raise ValueError(
                "Depth wrapper contract drifted. "
                f"Expected depth chain {expected_depth_chain}, got {active_depth_chain}."
            )

    return tuple(wrapper_order)


def build_env(
    task: str,
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    headless: bool,
    video: VideoCfg | None,
    video_mode: Literal["train", "play"],
    log_dir: str | None,
    wrappers: EnvWrapperCfg,
) -> VecEnvWrapper:
    import gymnasium as gym
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

    from dawn.sim.utils.env_wrappers import (
        CameraFollowWrapper,
        ChannelFirstWrapper,
        DepthBufferWrapper,
        DepthNoiseWrapper,
        EnvIndexWrapper,
        IsaaclabWrapper,
        ObservationWrapper,
        VisibilityWrapper,
        VisualizationWrapper,
    )
    from dawn.utils.video_utils import record_video, record_video_play

    enable_video = video is not None and video.enable
    has_depth = env_uses_depth_camera(env_cfg) and getattr(env_cfg, "depth", None) is not None

    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if enable_video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if enable_video:
        if video_mode == "train":
            env = record_video(env, log_dir, video.length, video.interval)
        else:
            env = record_video_play(env, log_dir, video.length)

    wrapper_order = get_runtime_wrapper_order(has_depth=has_depth, headless=headless, wrappers=wrappers)
    for wrapper_name in wrapper_order:
        if wrapper_name == "IsaaclabWrapper":
            env = IsaaclabWrapper(env)
        elif wrapper_name == "ChannelFirstWrapper":
            env = ChannelFirstWrapper(env)
        elif wrapper_name == "DepthBufferWrapper":
            env = DepthBufferWrapper(env)
        elif wrapper_name == "DepthNoiseWrapper":
            env = DepthNoiseWrapper(env)
        elif wrapper_name == "ObservationWrapper":
            env = ObservationWrapper(env)
        elif wrapper_name == "VisibilityWrapper":
            env = VisibilityWrapper(env)
        elif wrapper_name == "EnvIndexWrapper":
            env = EnvIndexWrapper(env)
        elif wrapper_name == "CameraFollowWrapper":
            env = CameraFollowWrapper(env)
        elif wrapper_name == "VisualizationWrapper":
            env = VisualizationWrapper(env)

    env.reset()
    return env
