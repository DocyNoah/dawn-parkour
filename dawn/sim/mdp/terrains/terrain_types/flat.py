from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dawn.sim.mdp.terrains.sub_terrains_cfg import HfRandomUniformTerrainCfg


def flat_terrain_height_field(difficulty: float, cfg: HfRandomUniformTerrainCfg) -> np.ndarray:
    x_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    y_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    hf_raw = np.zeros((x_pixels, y_pixels))

    hf_raw[:, :] = 0

    return np.rint(hf_raw).astype(np.int16)
