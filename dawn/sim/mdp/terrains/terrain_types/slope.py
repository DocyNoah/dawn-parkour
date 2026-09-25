from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dawn.sim.mdp.terrains.sub_terrains_cfg import HfPyramidSlopedTerrainCfg


def pyramid_sloped_terrain_height_field(difficulty: float, cfg: HfPyramidSlopedTerrainCfg) -> np.ndarray:
    slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    if cfg.inverted:
        slope *= -1

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_max = int(slope * cfg.size[0] / 2 / cfg.vertical_scale)
    center_x = width_pixels // 2
    center_y = length_pixels // 2

    x = np.arange(width_pixels)
    y = np.arange(length_pixels)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y
    xx = xx.reshape(width_pixels, 1)
    yy = yy.reshape(1, length_pixels)
    hf_raw = height_max * xx * yy

    platform_width = int(cfg.platform_width / cfg.horizontal_scale / 2)
    x_pf = center_x - platform_width
    y_pf = center_y - platform_width
    z_pf = hf_raw[x_pf, y_pf]
    hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))

    return np.rint(hf_raw).astype(np.int16)
