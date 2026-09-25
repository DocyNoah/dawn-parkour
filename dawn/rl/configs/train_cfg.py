from __future__ import annotations

import glob
from dataclasses import field
from pathlib import Path
from typing import Any, Literal

from isaaclab.utils import configclass

from dawn.rl.configs.world_model_cfg import DAWNWorldModelCfg


@configclass
class LoggingCfg:
    experiment_name: str = "DAWN"

    run_name: str = ""

    now_time: str = ""

    logger: Literal["tensorboard", "wandb"] = "tensorboard"

    wandb_project: str = "dawn-parkour"


@configclass
class TrainRunnerCfg:
    num_steps_per_env: int = 24

    max_iterations: int = 10_000

    save_interval: int = 50


@configclass
class ActorCriticPolicyCfg:
    actor_network_name: str = ""

    actor_network_cfg: dict[str, Any] = field(default_factory=dict)

    critic_network_name: str = ""

    critic_network_cfg: dict[str, Any] = field(default_factory=dict)

    init_noise_std: float = 1.0

    noise_std_type: str = "scalar"

    is_recurrent: bool = False

    actor_obs_normalization: bool = False

    critic_obs_normalization: bool = False


@configclass
class PPOAlgorithmCfg:
    value_loss_coef: float = 1.0

    use_clipped_value_loss: bool = True

    clip_param: float = 0.2

    entropy_coef: float = 0.01

    num_learning_epochs: int = 5

    num_mini_batches: int = 4

    learning_rate: float = 1e-3

    schedule: str = "adaptive"

    gamma: float = 0.99

    lam: float = 0.95

    desired_kl: float = 0.01

    max_grad_norm: float = 1.0

    normalize_advantage_per_mini_batch: bool = False


@configclass
class PPOAMPAlgorithmCfg(PPOAlgorithmCfg):
    amp_replay_buffer_size: int = 1_000_000

    vel_predict_coef: float = 1.0


@configclass
class AMPCfg:
    amp_reward_coef: float = 0.5 * 0.02

    amp_motion_files: list[str] = field(
        default_factory=lambda: glob.glob(str(Path(__file__).resolve().parents[2] / "datasets/mocap_motions/*.txt"))
    )

    amp_num_preload_transitions: int = 2_000_000

    amp_task_reward_lerp: float = 0.3

    amp_hidden_layer_sizes: list[int] = field(default_factory=lambda: [1024, 512])

    min_normalized_std: list[float] = field(default_factory=lambda: [0.05] * 4 + [0.02] * 4 + [0.05] * 4)


@configclass
class DepthPredictorCfg:
    cnn_base_channels: int = 32

    lr: float = 3e-4

    weight_decay: float = 1e-4

    train_interval: int = 10

    training_iters: int = 1000

    batch_size: int = 1024

    loss_scale: int = 100


@configclass
class DawnTrainCfg:
    algorithm_name: Literal["dawn"] = "dawn"

    runner: TrainRunnerCfg = field(default_factory=TrainRunnerCfg)

    policy: ActorCriticPolicyCfg = field(default_factory=ActorCriticPolicyCfg)

    algorithm: PPOAMPAlgorithmCfg = field(default_factory=PPOAMPAlgorithmCfg)

    amp: AMPCfg = field(default_factory=AMPCfg)

    wm: DAWNWorldModelCfg = field(default_factory=DAWNWorldModelCfg)

    depth_predictor: DepthPredictorCfg = field(default_factory=DepthPredictorCfg)
