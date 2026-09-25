import gymnasium as gym
from isaaclab.utils import configclass

from dawn.sim.envs.dawn.go1_dawn_root import Go1DawnRootEnvCfg, set_play_config
from dawn.sim.envs.manager_based_rl_amp_env import ManagerBasedRLAMPEnv
from dawn.sim.mdp.observations.observation_groups import DawnObservationCfg


@configclass
class DawnEnvCfg(Go1DawnRootEnvCfg):
    observations: DawnObservationCfg = DawnObservationCfg()


@configclass
class DawnPlayEnvCfg(DawnEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        set_play_config(self)


for task_name, cfg_type in (
    ("Go1-DAWN-v0", DawnEnvCfg),
    ("Go1-DAWN-Play-v0", DawnPlayEnvCfg),
):
    gym.register(
        id=task_name,
        entry_point=ManagerBasedRLAMPEnv,
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": cfg_type},
    )
