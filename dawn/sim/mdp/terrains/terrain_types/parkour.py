from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dawn.sim.mdp.terrains.sub_terrains_cfg import HfParkourTerrainCfg


def parkour_terrain_height_field(difficulty: float, cfg: HfParkourTerrainCfg) -> np.ndarray:
    depth = int(difficulty * 0.54 / cfg.vertical_scale)

    x_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    y_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    hf_raw = np.zeros((x_pixels, y_pixels))

    x1 = int(2 / cfg.horizontal_scale)
    length = cfg.huddle_size + np.random.uniform(-0.05, 0.05)
    x2 = x1 + int(length / cfg.horizontal_scale)

    x3 = int(6 / cfg.horizontal_scale)
    x4 = x3 + int(length / cfg.horizontal_scale)
    hf_raw[x1:x2, :] = depth
    hf_raw[x3:x4, :] = depth

    return np.rint(hf_raw).astype(np.int16)
