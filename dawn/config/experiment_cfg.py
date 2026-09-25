from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from dawn.rl.configs.train_cfg import (
    ActorCriticPolicyCfg,
    AMPCfg,
    DawnTrainCfg,
    DepthPredictorCfg,
    LoggingCfg,
    PPOAlgorithmCfg,
    PPOAMPAlgorithmCfg,
    TrainRunnerCfg,
)


@dataclass
class AppCfg:
    headless: bool = True

    enable_cameras: bool = False

    device: str = "cuda:0"

    livestream: Literal[0, 1, 2] = 0

    xr: bool = False

    verbose: bool = False

    info: bool = False

    experience: str = ""

    rendering_mode: Literal["performance", "balanced", "quality"] | None = None

    kit_args: str = ""

    def to_app_launcher_dict(self) -> dict:
        from dataclasses import asdict

        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class EnvCfg:
    task: str | None = None

    seed: int = -1

    num_envs: int | None = None

    camera_num_envs: int | None = None

    depth_image_shape: tuple[int, int] | None = None

    invisible_robot: bool = False


@dataclass
class VideoCfg:
    enable: bool = False

    length: int = 200

    interval: int = 2000


@dataclass
class TorchBackendCfg:
    cuda_matmul_allow_tf32: bool = True

    cudnn_allow_tf32: bool = True

    cudnn_deterministic: bool = False

    cudnn_benchmark: bool = False


@dataclass
class EnvWrapperCfg:
    use_observation_wrapper: bool = True

    use_depth_buffer_wrapper: bool = False

    use_depth_noise_wrapper: bool = False

    use_visibility_wrapper: bool = False

    use_camera_follow_wrapper: bool = False

    use_visualization_wrapper: bool = False


@dataclass
class ExperimentCfg:
    app: AppCfg = field(default_factory=AppCfg)

    env: EnvCfg = field(default_factory=EnvCfg)

    video: VideoCfg = field(default_factory=VideoCfg)

    torch_backend: TorchBackendCfg = field(default_factory=TorchBackendCfg)

    wrappers: EnvWrapperCfg = field(default_factory=EnvWrapperCfg)

    train: DawnTrainCfg = field(default_factory=DawnTrainCfg)

    logging: LoggingCfg = field(default_factory=LoggingCfg)


__all__ = [
    "AMPCfg",
    "ActorCriticPolicyCfg",
    "AppCfg",
    "DawnTrainCfg",
    "DepthPredictorCfg",
    "EnvCfg",
    "EnvWrapperCfg",
    "ExperimentCfg",
    "LoggingCfg",
    "PPOAMPAlgorithmCfg",
    "PPOAlgorithmCfg",
    "TorchBackendCfg",
    "TrainRunnerCfg",
    "VideoCfg",
]
