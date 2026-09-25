from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dawn.sim.mdp.terrains.sub_terrains_cfg import HfStepTerrainCfg


def step_center_terrain_height_field(difficulty: float, cfg: HfStepTerrainCfg) -> np.ndarray:
    height = difficulty * 0.54
    depth = int(height / cfg.vertical_scale)
    if cfg.play:
        depth = int(cfg.play_step_size / cfg.vertical_scale)

    x_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    y_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    hf_raw = np.zeros((x_pixels, y_pixels))

    center_y = y_pixels // 2
    width = 0.9 + 0.1 * np.random.uniform(-1, 1)
    half_width = width / 2
    y1 = center_y - int(half_width / cfg.horizontal_scale)
    y2 = center_y + int(half_width / cfg.horizontal_scale)

    x1 = int(1 / cfg.horizontal_scale)
    length = 1.0 + 0.2 * np.random.random()
    x2 = int((1 + length) / cfg.horizontal_scale)
    x3 = int(5 / cfg.horizontal_scale)
    length = 1.0 + 0.2 * np.random.random()
    x4 = int((5 + length) / cfg.horizontal_scale)
    if cfg.play:
        x4 = int(8 / cfg.horizontal_scale)

    hf_raw[x1:x2, y1:y2] = depth
    hf_raw[x3:x4, y1:y2] = depth

    return np.rint(hf_raw).astype(np.int16)
