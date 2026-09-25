import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity import mdp

from dawn.sim.mdp.events import (
    randomize_camera_offset,
    randomize_obs_delay_steps,
    randomize_rigid_body_com,
    randomize_rigid_body_material,
)


@configclass
class EventsPreset:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "mass_distribution_params": (1.0, 5.0),
            "operation": "add",
        },
    )

    randomize_rigid_body_com = EventTerm(
        func=randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "com_range": {
                "x": (-0.1, 0.1),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.2),
            },
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.75, 0.75),
                "y": (-0.5, 0.5),
                "yaw": (-math.pi, math.pi),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    ground_material = EventTerm(
        func=randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot"),
            "friction_range": (0.4, 1.2),
            "friction_scale_factor": (1.1, 1.5),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 100_000,
        },
    )

    randomize_joint_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.0, 0.05),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    randomize_obs_delay_steps = EventTerm(
        func=randomize_obs_delay_steps,
        mode="reset",
        params={
            "obs_delay_range_steps": (0, 2),
        },
    )

    randomize_camera_offset = EventTerm(
        func=randomize_camera_offset,
        mode="reset",
        params={
            "sensor_cfg": SceneEntityCfg("depth_camera"),
            "position_range": {
                "x": (0.30, 0.30),
                "y": (0.0, 0.0),
                "z": (0.12, 0.12),
            },
            "orientation_range": {
                "roll": (0.0, 0.0),
                "pitch": (25 * math.pi / 180, 35 * math.pi / 180),
                "yaw": (0.0, 0.0),
            },
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
