from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity import mdp

from dawn.sim.mdp.observations import observations as obs

MODEL_HISTORY_LENGTH = 5
MAX_OBS_DELAY_STEP = 2
history_length_with_delay = MODEL_HISTORY_LENGTH + MAX_OBS_DELAY_STEP


@configclass
class PrivCfg(ObsGroup):
    contact_flag = ObsTerm(
        func=obs.contact_flag,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_calf", ".*_thigh"])},
    )
    contact_footforce = ObsTerm(
        func=obs.contact_footforce,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"])},
        scale=0.005,
    )
    p_gain = ObsTerm(func=obs.p_gain, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}, scale=5.0)
    d_gain = ObsTerm(func=obs.d_gain, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}, scale=5.0)
    mass = ObsTerm(func=obs.mass)
    com = ObsTerm(
        func=obs.com_pos_b,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="trunk")},
        scale=20.0,
    )
    friction = ObsTerm(func=obs.friction)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=1.0)


@configclass
class PropCleanCfg(ObsGroup):
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25)
    orientation = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), scale=1.0)
    joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.05)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class PropHistoryCfg(PropCleanCfg):
    history_length = history_length_with_delay
    flatten_history_dim = False

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class CommandCfg(ObsGroup):
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class LastActionCfg(ObsGroup):
    last_action = ObsTerm(func=mdp.last_action, clip=(-6.0, 6.0))

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class LastActionHistoryCfg(LastActionCfg):
    history_length = MODEL_HISTORY_LENGTH
    flatten_history_dim = False

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ScanCfg(ObsGroup):
    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.3},
        clip=(-1.0, 1.0),
        scale=5.0,
    )


@configclass
class ForwardHeightMapCfg(ObsGroup):
    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("forward_height_scanner")},
        clip=(-1.0, 1.0),
        scale=5.0,
    )


@configclass
class DepthCfg(ObsGroup):
    depth_image = ObsTerm(
        func=obs.image,
        params={
            "sensor_cfg": SceneEntityCfg("depth_camera"),
            "data_type": "depth",
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class AMPObsCfg(ObsGroup):
    joint_pos = ObsTerm(
        func=obs.joint_pos_amp,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    joint_vel = ObsTerm(
        func=obs.joint_vel_amp,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
