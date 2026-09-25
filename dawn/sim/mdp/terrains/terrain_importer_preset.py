import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from dawn.sim.mdp.terrains.terrain_importer_cfg import TerrainImporterWithEdgesCfg


def get_terrain_cfg(terrain_generator: TerrainGeneratorCfg) -> TerrainImporterWithEdgesCfg:
    return TerrainImporterWithEdgesCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_generator,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.3,
            dynamic_friction=1.6,
            restitution=0.5,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
