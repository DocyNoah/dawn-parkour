from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dawn.sim.mdp.terrains.sub_terrains_cfg import HfPyramidStairsTerrainCfg


def pyramid_stairs_terrain_height_field(difficulty: float, cfg: HfPyramidStairsTerrainCfg) -> np.ndarray:
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    if cfg.inverted:
        step_height *= -1

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    step_width = int(cfg.step_width / cfg.horizontal_scale)
    step_height = int(step_height / cfg.vertical_scale)
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)
    border_pixels = int(cfg.border_width / cfg.horizontal_scale)
    sub_width_pixels = width_pixels - 2 * border_pixels
    sub_length_pixels = length_pixels - 2 * border_pixels
    sub_hf_raw = np.zeros((sub_width_pixels, sub_length_pixels))

    current_step_height = 0
    start_x, start_y = 0, 0
    stop_x, stop_y = sub_width_pixels, sub_length_pixels
    while (stop_x - start_x) > platform_width and (stop_y - start_y) > platform_width:
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        current_step_height += step_height
        sub_hf_raw[start_x:stop_x, start_y:stop_y] = current_step_height

    hf_raw = np.zeros((width_pixels, length_pixels))
    hf_raw[border_pixels:-border_pixels, border_pixels:-border_pixels] = sub_hf_raw

    return np.rint(hf_raw).astype(np.int16)
