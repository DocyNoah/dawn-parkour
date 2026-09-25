import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity import mdp

from dawn.sim.mdp.rewards import rewards


@configclass
class RewardsPreset:
    track_lin_vel_xy_exp = RewTerm(
        func=rewards.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.15, "lin_vel_clip": 0.1},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.15)}
    )

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)

    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0001)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.03)

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    stuck = RewTerm(func=rewards.stuck, weight=-1.0, params={"command_name": "base_velocity"})

    joint_deviation_l2 = RewTerm(func=rewards.joint_deviation_l2, weight=-0.04)

    collision = RewTerm(
        func=rewards.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 0.1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_calf", ".*_thigh"]),
        },
    )

    stumble = RewTerm(
        func=rewards.stumble,
        weight=-0.1,
        params={
            "ratio_threshold": 4.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )

    edge_avoidance = RewTerm(
        func=rewards.edge_avoidance,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )

    heading_deviation = RewTerm(func=rewards.heading_deviation, weight=-1.0)

    stand_still_contact_penalty = RewTerm(
        func=rewards.stand_still_contact_penalty,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "contact_threshold": 1.0,
            "command_threshold": 0.1,
        },
    )

    stand_still_default_pos_penalty = RewTerm(
        func=rewards.stand_still_default_pos_penalty,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "command_threshold": 0.1,
        },
    )

    off_center_penalty = RewTerm(
        func=rewards.off_center_penalty,
        weight=-0.01,
        params={
            "safe_distance": 0.3,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
