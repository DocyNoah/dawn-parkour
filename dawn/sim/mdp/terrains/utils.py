from __future__ import annotations

import copy
import functools
from typing import TYPE_CHECKING

import numpy as np
import trimesh

if TYPE_CHECKING:
    from collections.abc import Callable

    from isaaclab.terrains import HfTerrainBaseCfg


def height_field_to_mesh_with_edges(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(difficulty: float, cfg: HfTerrainBaseCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
        if cfg.border_width > 0 and cfg.border_width < cfg.horizontal_scale:
            raise ValueError(
                f"The border width ({cfg.border_width}) must be greater than or equal to the"
                f" horizontal scale ({cfg.horizontal_scale})."
            )

        width_pixels = int(cfg.size[0] / cfg.horizontal_scale) + 1
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale) + 1
        border_pixels = int(cfg.border_width / cfg.horizontal_scale) + 1
        heights = np.zeros((width_pixels, length_pixels), dtype=np.int16)

        sub_terrain_size = [width_pixels - 2 * border_pixels, length_pixels - 2 * border_pixels]
        sub_terrain_size = [dim * cfg.horizontal_scale for dim in sub_terrain_size]

        terrain_size = copy.deepcopy(cfg.size)
        cfg.size = tuple(sub_terrain_size)

        z_gen = func(difficulty, cfg)

        heights[border_pixels:-border_pixels, border_pixels:-border_pixels] = z_gen

        cfg.size = terrain_size

        vertices, triangles, is_edge_x, is_edge_y, is_edge_c = convert_height_field_to_mesh(
            heights, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

        x1 = int((0.5 * cfg.size[0] - 1) / cfg.horizontal_scale)
        x2 = int((0.5 * cfg.size[0] + 1) / cfg.horizontal_scale)
        y1 = int((0.5 * cfg.size[1] - 1) / cfg.horizontal_scale)
        y2 = int((0.5 * cfg.size[1] + 1) / cfg.horizontal_scale)
        origin_z = np.max(heights[x1:x2, y1:y2]) * cfg.vertical_scale
        origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], origin_z])

        return [mesh], origin, is_edge_x, is_edge_y, is_edge_c

    return wrapper


def convert_height_field_to_mesh(
    height_field: np.ndarray, horizontal_scale: float, vertical_scale: float, slope_threshold: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_rows, num_cols = height_field.shape

    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    hf = height_field.copy()

    if slope_threshold is not None:
        slope_threshold *= horizontal_scale / vertical_scale

        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))

        move_x[: num_rows - 1, :] += hf[1:num_rows, :] - hf[: num_rows - 1, :] > slope_threshold
        move_x[1:num_rows, :] -= hf[: num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold

        move_y[:, : num_cols - 1] += hf[:, 1:num_cols] - hf[:, : num_cols - 1] > slope_threshold
        move_y[:, 1:num_cols] -= hf[:, : num_cols - 1] - hf[:, 1:num_cols] > slope_threshold

        move_corners[: num_rows - 1, : num_cols - 1] += (
            hf[1:num_rows, 1:num_cols] - hf[: num_rows - 1, : num_cols - 1] > slope_threshold
        )
        move_corners[1:num_rows, 1:num_cols] -= (
            hf[: num_rows - 1, : num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold
        )
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale

    triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1 : stop : 2, 0] = ind0
        triangles[start + 1 : stop : 2, 1] = ind2
        triangles[start + 1 : stop : 2, 2] = ind3

    return vertices, triangles, move_x != 0, move_y != 0, move_corners != 0
