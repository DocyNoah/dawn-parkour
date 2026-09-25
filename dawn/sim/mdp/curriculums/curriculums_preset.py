from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from dawn.sim.mdp.curriculums import curriculums


@configclass
class CurriculumsPreset:
    terrain_levels = CurrTerm(func=curriculums.terrain_levels_vel_stats)
