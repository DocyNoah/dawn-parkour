from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.interpolate as interpolate

if TYPE_CHECKING:
    from dawn.sim.mdp.terrains.sub_terrains_cfg import HfRandomUniformTerrainCfg


def random_uniform_terrain_height_field(difficulty: float, cfg: HfRandomUniformTerrainCfg) -> np.ndarray:
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale
    elif cfg.downsampled_scale < cfg.horizontal_scale:
        raise ValueError(
            "Downsampled scale must be larger than or equal to the horizontal scale:"
            f" {cfg.downsampled_scale} < {cfg.horizontal_scale}."
        )

    max_height_variation = (
        cfg.height_variation_range[1] - cfg.height_variation_range[0]
    ) * difficulty + cfg.height_variation_range[0]

    if cfg.randomize_height_variation:
        height_variation = np.random.uniform(cfg.height_variation_range[0], max_height_variation)
    else:
        height_variation = max_height_variation

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)

    height_min = int(-height_variation / cfg.vertical_scale)
    height_max = int(height_variation / cfg.vertical_scale)
    height_step = int(cfg.height_step / cfg.vertical_scale)

    height_range = np.arange(height_min, height_max + height_step, height_step)

    height_field_downsampled = np.random.choice(height_range, size=(width_downsampled, length_downsampled))

    x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)

    x_upsampled = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)

    return np.rint(z_upsampled).astype(np.int16)
