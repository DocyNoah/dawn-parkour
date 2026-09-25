from isaaclab.utils import configclass

from dawn.sim.mdp.observations.observation_terms import (
    AMPObsCfg,
    CommandCfg,
    DepthCfg,
    ForwardHeightMapCfg,
    LastActionHistoryCfg,
    PrivCfg,
    PropCleanCfg,
    PropHistoryCfg,
    ScanCfg,
)


@configclass
class DawnObservationCfg:
    scan: ScanCfg = ScanCfg()
    priv: PrivCfg = PrivCfg()
    prop_clean: PropCleanCfg = PropCleanCfg()
    prop_hist: PropHistoryCfg = PropHistoryCfg()
    action_hist: LastActionHistoryCfg = LastActionHistoryCfg()
    depth: DepthCfg = DepthCfg()
    command: CommandCfg = CommandCfg()
    amp_obs: AMPObsCfg = AMPObsCfg()
    forward_height_map: ForwardHeightMapCfg = ForwardHeightMapCfg()
