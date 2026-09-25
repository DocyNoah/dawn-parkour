from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG


def get_robot_cfg() -> ArticulationCfg:
    go1_cfg = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    go1_cfg.actuators["base_legs"] = DCMotorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.7,
        saturation_effort=23.7,
        velocity_limit=30.0,
        damping=None,
        stiffness=None,
        friction=0.0,
    )
    return go1_cfg
