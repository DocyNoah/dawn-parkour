from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dawn.sim.mdp.terrains.sub_terrains_cfg import HfGapTerrainCfg


def gap_center_terrain_height_field(difficulty: float, cfg: HfGapTerrainCfg) -> np.ndarray:
    gap_size = int(difficulty * 0.9 / cfg.horizontal_scale)
    if cfg.play:
        gap_size = int(cfg.play_gap_size / cfg.horizontal_scale)

    x_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    y_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    hf_raw = np.zeros((x_pixels, y_pixels))

    center_x = x_pixels // 2
    center_y = y_pixels // 2
    push = int(0.1 / cfg.horizontal_scale)

    x1 = center_x - int(1 / cfg.horizontal_scale) - push
    x2 = center_x + int(2 / cfg.horizontal_scale) - push
    x3 = x1 - gap_size - push
    x4 = x2 + gap_size

    width = 0.9 + 0.05 * np.random.uniform(-1, 1)
    half_width = width / 2
    y1 = center_y - int(half_width / cfg.horizontal_scale)
    y2 = center_y + int(half_width / cfg.horizontal_scale)

    x5 = gap_size - push
    x6 = x_pixels - push
    depth = int(np.random.uniform(0.27, 0.33) / cfg.vertical_scale)

    hf_raw[x5:x3, y1:y2] = depth
    hf_raw[x1:x2, y1:y2] = depth
    hf_raw[x4:x6, y1:y2] = depth

    return np.rint(hf_raw).astype(np.int16)
