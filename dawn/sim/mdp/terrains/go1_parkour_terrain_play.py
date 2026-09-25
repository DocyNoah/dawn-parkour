from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from dawn.sim.mdp.terrains.sub_terrains_cfg import HfParkourTerrainCfg, HfStepTerrainCfg

terrain_generator_cfg = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=15.0,
    num_rows=10,
    num_cols=4,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.5,
    use_cache=False,
    sub_terrains={
        "step_0.2": HfStepTerrainCfg(
            proportion=0.1,
            border_width=0.0,
            play=True,
            play_step_size=0.20,
            add_random_uniform_terrain=True,
            height_variation_range=(0.01, 0.01),
        ),
        "huddle_0.6": HfParkourTerrainCfg(
            proportion=0.1,
            border_width=0.0,
            height_variation_range=(0.01, 0.01),
            huddle_size=0.6,
        ),
        "huddle_0.7": HfParkourTerrainCfg(
            proportion=0.1,
            border_width=0.0,
            height_variation_range=(0.01, 0.01),
            huddle_size=0.7,
        ),
        "huddle_0.8": HfParkourTerrainCfg(
            proportion=0.1,
            border_width=0.0,
            height_variation_range=(0.01, 0.01),
            huddle_size=0.8,
        ),
    },
)
