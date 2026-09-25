from __future__ import annotations

from dataclasses import MISSING

from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.utils import configclass

from dawn.sim.mdp.terrains.sub_terrains import (
    flat_terrain,
    gap_center_terrain,
    parkour_terrain,
    pyramid_sloped_terrain,
    pyramid_stairs_terrain,
    step_center_terrain,
)


@configclass
class HfRandomUniformTerrainCfg(HfTerrainBaseCfg):
    function = flat_terrain

    randomize_height_variation: bool = True

    height_variation_range: tuple[float, float] = (0.02, 0.05)

    height_step: float = 0.005

    downsampled_scale: float | None = 0.10

    add_random_uniform_terrain: bool = MISSING


@configclass
class HfStepTerrainCfg(HfRandomUniformTerrainCfg):
    function = step_center_terrain

    num_steps: int = 3

    step_height_range_easy: tuple[float, float] = (0.1, 0.2)

    step_height_range_hard: tuple[float, float] | None = (0.15, 0.40)

    step_width_range: tuple[float, float] = (0.8, 1.6)

    step_depth_range: tuple[float, float] = (0.1, 0.4)

    use_curriculum_step_depth: bool = True

    step_dis_range: tuple[float, float] = (1.5, 2.0)

    add_random_uniform_terrain: bool = True

    play: bool = False

    play_step_size: float | None = None


@configclass
class HfGapTerrainCfg(HfRandomUniformTerrainCfg):
    function = gap_center_terrain

    num_gaps: int = 3

    gap_height_range: tuple[float, float] = (0.2, 1.0)

    gap_size_range: tuple[float, float] = (0.1, 0.8)

    gap_width_range: tuple[float, float] = (1.2, 2.0)

    stone_depth_range: tuple[float, float] = (0.8, 1.6)

    add_random_uniform_terrain: bool = True

    play: bool = False

    play_gap_size: float | None = None


@configclass
class HfParkourTerrainCfg(HfRandomUniformTerrainCfg):
    function = parkour_terrain

    num_rows: int = 10

    use_wall: bool = False

    add_random_uniform_terrain: bool = True

    play: bool = False

    play_step_size: float | None = None

    huddle_size: float = 0.6


@configclass
class HfPyramidStairsTerrainCfg(HfRandomUniformTerrainCfg):
    function = pyramid_stairs_terrain

    step_height_range: tuple[float, float] = MISSING

    step_width: float = MISSING

    platform_width: float = 1.0

    inverted: bool = False

    add_random_uniform_terrain: bool = True


@configclass
class HfInvertedPyramidStairsTerrainCfg(HfPyramidStairsTerrainCfg):
    inverted: bool = True


@configclass
class HfPyramidSlopedTerrainCfg(HfRandomUniformTerrainCfg):
    function = pyramid_sloped_terrain

    slope_range: tuple[float, float] = MISSING

    platform_width: float = 1.0

    inverted: bool = False
