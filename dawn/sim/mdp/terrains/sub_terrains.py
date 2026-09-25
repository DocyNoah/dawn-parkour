from __future__ import annotations

from typing import TYPE_CHECKING

from dawn.sim.mdp.terrains.terrain_types import (
    flat_terrain_height_field,
    gap_center_terrain_height_field,
    parkour_terrain_height_field,
    pyramid_sloped_terrain_height_field,
    pyramid_stairs_terrain_height_field,
    random_uniform_terrain_height_field,
    step_center_terrain_height_field,
)
from dawn.sim.mdp.terrains.utils import height_field_to_mesh_with_edges

if TYPE_CHECKING:
    import numpy as np

    from dawn.sim.mdp.terrains.sub_terrains_cfg import (
        HfGapTerrainCfg,
        HfParkourTerrainCfg,
        HfPyramidSlopedTerrainCfg,
        HfPyramidStairsTerrainCfg,
        HfRandomUniformTerrainCfg,
        HfStepTerrainCfg,
    )


@height_field_to_mesh_with_edges
def flat_terrain(difficulty: float, cfg: HfRandomUniformTerrainCfg) -> np.ndarray:
    hf_raw = flat_terrain_height_field(difficulty, cfg)
    if cfg.add_random_uniform_terrain:
        hf_raw += random_uniform_terrain_height_field(difficulty, cfg)
    return hf_raw


@height_field_to_mesh_with_edges
def step_center_terrain(difficulty: float, cfg: HfStepTerrainCfg) -> np.ndarray:
    hf_raw = step_center_terrain_height_field(difficulty, cfg)
    if cfg.add_random_uniform_terrain:
        hf_raw += random_uniform_terrain_height_field(difficulty, cfg)
    return hf_raw


@height_field_to_mesh_with_edges
def gap_center_terrain(difficulty: float, cfg: HfGapTerrainCfg) -> np.ndarray:
    hf_raw = gap_center_terrain_height_field(difficulty, cfg)
    if cfg.add_random_uniform_terrain:
        hf_raw += random_uniform_terrain_height_field(difficulty, cfg)
    return hf_raw


@height_field_to_mesh_with_edges
def parkour_terrain(difficulty: float, cfg: HfParkourTerrainCfg) -> np.ndarray:
    difficulty = int(difficulty * 10) / 10
    hf_raw = parkour_terrain_height_field(difficulty, cfg)
    if cfg.add_random_uniform_terrain:
        hf_raw += random_uniform_terrain_height_field(difficulty, cfg)
    return hf_raw


@height_field_to_mesh_with_edges
def pyramid_stairs_terrain(difficulty: float, cfg: HfPyramidStairsTerrainCfg) -> np.ndarray:
    hf_raw = pyramid_stairs_terrain_height_field(difficulty, cfg)
    if cfg.add_random_uniform_terrain:
        hf_raw += random_uniform_terrain_height_field(difficulty, cfg)
    return hf_raw


@height_field_to_mesh_with_edges
def pyramid_sloped_terrain(difficulty: float, cfg: HfPyramidSlopedTerrainCfg) -> np.ndarray:
    hf_raw = pyramid_sloped_terrain_height_field(difficulty, cfg)
    if cfg.add_random_uniform_terrain:
        hf_raw += random_uniform_terrain_height_field(difficulty, cfg)
    return hf_raw
