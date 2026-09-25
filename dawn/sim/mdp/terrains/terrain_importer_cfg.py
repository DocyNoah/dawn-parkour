from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg

from dawn.sim.mdp.terrains.terrain_importer import TerrainImporterWithEdges


class TerrainImporterWithEdgesCfg(TerrainImporterCfg):
    edge_width_thresh: float = 0.05

    def __post_init__(self):
        super().__post_init__()
        self.class_type = TerrainImporterWithEdges
