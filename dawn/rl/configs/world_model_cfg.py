from dataclasses import field

from isaaclab.utils import configclass


@configclass
class WorldModelAlgorithmCfg:
    train_start_steps: int = 10_000

    train_steps_per_iter: int = 10

    batch_size: int = 16

    batch_length: int = 64

    micro_batch_size: int = 4


@configclass
class WorldModelOptimCfg:
    model_lr: float = 1e-4

    opt_eps: float = 1e-8

    grad_clip: float = 1000.0

    weight_decay: float = 0.0


@configclass
class WorldModelEncoderCfg:
    mlp_keys: str = ".*"

    cnn_keys: str = "image"

    act: str = "SiLU"

    norm: bool = True

    cnn_depth: int = 32

    kernel_size: int = 4

    minres: int = 4

    mlp_layers: int = 5

    mlp_units: int = 1024

    symlog_inputs: bool = True


@configclass
class WorldModelDecoderCfg:
    mlp_keys: str = ".*"

    cnn_keys: str = "image"

    act: str = "SiLU"

    norm: bool = True

    cnn_depth: int = 32

    kernel_size: int = 4

    minres: int = 4

    mlp_layers: int = 5

    mlp_units: int = 1024

    cnn_sigmoid: bool = False

    image_dist: str = "mse"

    vector_dist: str = "symlog_mse"

    outscale: float = 1.0


@configclass
class WorldModelRssmCfg:
    precision: int = 32

    dyn_hidden: int = 512

    dyn_deter: int = 512

    dyn_stoch: int = 32

    dyn_discrete: int = 32

    units: int = 512

    act: str = "SiLU"

    norm: bool = True

    dyn_scale: float = 0.5

    rep_scale: float = 0.1

    kl_free: float = 1.0

    unimix_ratio: float = 0.01


@configclass
class WorldModelCfg:
    algorithm: WorldModelAlgorithmCfg = field(default_factory=WorldModelAlgorithmCfg)

    optim: WorldModelOptimCfg = field(default_factory=WorldModelOptimCfg)

    rssm: WorldModelRssmCfg = field(default_factory=WorldModelRssmCfg)

    encoder: WorldModelEncoderCfg = field(default_factory=WorldModelEncoderCfg)

    decoder: WorldModelDecoderCfg = field(default_factory=WorldModelDecoderCfg)


@configclass
class WorldModelContrastiveCfg:
    enabled: bool = True

    projection_dim: int = 128

    projection_hidden: int = 256

    contrastive_coef: float = 0.1

    temperature: float = 0.07


@configclass
class DAWNWorldModelCfg(WorldModelCfg):
    contrastive: WorldModelContrastiveCfg = field(default_factory=WorldModelContrastiveCfg)


__all__ = [
    "DAWNWorldModelCfg",
    "WorldModelAlgorithmCfg",
    "WorldModelCfg",
    "WorldModelContrastiveCfg",
    "WorldModelDecoderCfg",
    "WorldModelEncoderCfg",
    "WorldModelOptimCfg",
    "WorldModelRssmCfg",
]
