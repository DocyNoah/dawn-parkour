import os
from typing import Literal

import numpy as np
import torch
import trimesh
from isaaclab.terrains import TerrainGenerator
from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils.dict import dict_to_md5_hash
from isaaclab.utils.io import dump_yaml


class TerrainGeneratorWithEdges(TerrainGenerator):
    terrain_mesh: trimesh.Trimesh

    terrain_meshes: list[trimesh.Trimesh]

    terrain_origins: np.ndarray

    terrain_edges_x: np.ndarray

    terrain_edges_y: np.ndarray

    terrain_edges_c: np.ndarray

    flat_patches: dict[str, torch.Tensor]

    _global_terrain_edges_x: np.ndarray

    _global_terrain_edges_y: np.ndarray

    _global_terrain_edges_c: np.ndarray

    def __init__(self, cfg: TerrainGeneratorCfg, device: str = "cpu"):
        self.is_flat_terrain = np.zeros(cfg.num_cols, dtype=bool)

        self.terrain_type_names = [""] * cfg.num_cols

        width_pixels = int(cfg.size[0] / cfg.horizontal_scale) + 1
        height_pixels = int(cfg.size[1] / cfg.horizontal_scale) + 1
        self.terrain_edges_x = np.zeros((cfg.num_rows, cfg.num_cols, width_pixels, height_pixels), dtype=bool)
        self.terrain_edges_y = np.zeros((cfg.num_rows, cfg.num_cols, width_pixels, height_pixels), dtype=bool)
        self.terrain_edges_c = np.zeros((cfg.num_rows, cfg.num_cols, width_pixels, height_pixels), dtype=bool)

        super().__init__(cfg, device)

        self._global_terrain_edges_x = self._compute_global_edge_map("x")
        self._global_terrain_edges_y = self._compute_global_edge_map("y")
        self._global_terrain_edges_c = self._compute_global_edge_map("c")

    def __str__(self):
        msg = super().__str__()

        edge_stats = self.get_all_edge_statistics()
        if edge_stats["total_terrains_with_edges"] > 0:
            msg += "\n\tEdge Statistics:"
            msg += f"\n\t\tTerrains with edges: {edge_stats['total_terrains_with_edges']}"
            msg += f"\n\t\tPercentage of terrains with edges: {edge_stats['percentage_with_edges']:.1f}%"
            msg += f"\n\t\tMean edges per terrain: {edge_stats['mean_edges']:.1f}"
            msg += f"\n\t\tMax edges in a terrain: {edge_stats['max_edges']}"
            msg += f"\n\t\tMin edges in a terrain: {edge_stats['min_edges']}"

        return msg

    def get_edge_info(self, row: int, col: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.terrain_edges_x[row, col].copy(),
            self.terrain_edges_y[row, col].copy(),
            self.terrain_edges_c[row, col].copy(),
        )

    def get_edge_count(self, row: int, col: int) -> dict[str, int]:
        is_edge_x, is_edge_y, is_edge_c = self.get_edge_info(row, col)
        return {
            "x_edges": int(np.sum(is_edge_x)),
            "y_edges": int(np.sum(is_edge_y)),
            "corner_edges": int(np.sum(is_edge_c)),
            "total_edges": int(np.sum(is_edge_x) + np.sum(is_edge_y) + np.sum(is_edge_c)),
        }

    def get_all_edge_statistics(self) -> dict[str, float]:
        edge_counts = []
        terrains_with_edges = 0

        for row in range(self.cfg.num_rows):
            for col in range(self.cfg.num_cols):
                edge_info = self.get_edge_count(row, col)
                total_edges = edge_info["total_edges"]
                edge_counts.append(total_edges)
                if total_edges > 0:
                    terrains_with_edges += 1

        return {
            "mean_edges": float(np.mean(edge_counts)),
            "max_edges": int(np.max(edge_counts)),
            "min_edges": int(np.min(edge_counts)),
            "total_terrains_with_edges": terrains_with_edges,
            "percentage_with_edges": float(terrains_with_edges / len(edge_counts) * 100),
        }

    def _compute_global_edge_map(self, axis: Literal["x", "y", "c"]) -> np.ndarray:
        sub_edge_source = {
            "x": self.terrain_edges_x,
            "y": self.terrain_edges_y,
            "c": self.terrain_edges_c,
        }[axis]

        total_width = self.cfg.num_rows * self.cfg.size[0]
        total_height = self.cfg.num_cols * self.cfg.size[1]

        global_width = int(total_width / self.cfg.horizontal_scale) + 1
        global_height = int(total_height / self.cfg.horizontal_scale) + 1

        sub_width_pixels = int(self.cfg.size[0] / self.cfg.horizontal_scale)
        sub_height_pixels = int(self.cfg.size[1] / self.cfg.horizontal_scale)

        global_edge_map = np.zeros((global_width, global_height), dtype=bool)

        for row in range(self.cfg.num_rows):
            for col in range(self.cfg.num_cols):
                start_x = row * sub_width_pixels
                end_x = start_x + sub_width_pixels + 1
                start_y = col * sub_height_pixels
                end_y = start_y + sub_height_pixels + 1

                end_x = min(end_x, global_width)
                end_y = min(end_y, global_height)

                sub_edge_data = sub_edge_source[row, col]
                actual_sub_width = end_x - start_x
                actual_sub_height = end_y - start_y

                global_edge_map[start_x:end_x, start_y:end_y] = sub_edge_data[:actual_sub_width, :actual_sub_height]

        return global_edge_map

    @property
    def global_terrain_edges_x(self) -> np.ndarray:
        return self._global_terrain_edges_x.copy()

    @property
    def global_terrain_edges_y(self) -> np.ndarray:
        return self._global_terrain_edges_y.copy()

    @property
    def global_terrain_edges_c(self) -> np.ndarray:
        return self._global_terrain_edges_c.copy()

    def _generate_random_terrains(self) -> None:
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)

        sub_terrain_names, sub_terrains_cfgs = zip(*self.cfg.sub_terrains.items())

        for index in range(self.cfg.num_rows * self.cfg.num_cols):
            (sub_row, sub_col) = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))

            sub_index = self.np_rng.choice(len(proportions), p=proportions)

            if self.terrain_type_names[sub_col] == "":
                picked_name = sub_terrain_names[sub_index]
                self.terrain_type_names[sub_col] = picked_name
                if "flat" in picked_name:
                    self.is_flat_terrain[sub_col] = True

            difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)

            mesh, origin, is_edge_x, is_edge_y, is_edge_c = self._get_terrain_mesh(
                difficulty, sub_terrains_cfgs[sub_index]
            )

            self._add_sub_terrain(
                mesh,
                origin,
                sub_row,
                sub_col,
                sub_terrains_cfgs[sub_index],
                is_edge_x,
                is_edge_y,
                is_edge_c,
            )

        for col in range(self.cfg.num_cols):
            if self.terrain_type_names[col] == "":
                self.terrain_type_names[col] = "unknown"
        self.terrain_type_names = np.array(self.terrain_type_names)

    def _generate_curriculum_terrains(self) -> None:
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)

        sub_indices = []
        for index in range(self.cfg.num_cols):
            sub_index = np.min(np.where(index / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
            sub_indices.append(sub_index)
        sub_indices = np.array(sub_indices, dtype=np.int32)

        sub_terrain_names, sub_terrains_cfgs = zip(*self.cfg.sub_terrains.items())

        for col in range(self.cfg.num_cols):
            terrain_type_idx = sub_indices[col]
            self.terrain_type_names[col] = sub_terrain_names[terrain_type_idx]
        self.terrain_type_names = np.array(self.terrain_type_names)

        for col_idx, sub_terrain_name in enumerate(sub_terrain_names):
            if "flat" in sub_terrain_name:
                cols_with_this_terrain = np.where(sub_indices == col_idx)[0]
                self.is_flat_terrain[cols_with_this_terrain] = True

        for sub_col in range(self.cfg.num_cols):
            for sub_row in range(self.cfg.num_rows):
                lower, upper = self.cfg.difficulty_range
                difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
                difficulty = lower + (upper - lower) * difficulty

                mesh, origin, is_edge_x, is_edge_y, is_edge_c = self._get_terrain_mesh(
                    difficulty, sub_terrains_cfgs[sub_indices[sub_col]]
                )

                self._add_sub_terrain(
                    mesh,
                    origin,
                    sub_row,
                    sub_col,
                    sub_terrains_cfgs[sub_indices[sub_col]],
                    is_edge_x,
                    is_edge_y,
                    is_edge_c,
                )

    def _add_sub_terrain(
        self,
        mesh: trimesh.Trimesh,
        origin: np.ndarray,
        row: int,
        col: int,
        sub_terrain_cfg: SubTerrainBaseCfg,
        is_edge_x: np.ndarray,
        is_edge_y: np.ndarray,
        is_edge_c: np.ndarray,
    ) -> None:
        self.terrain_edges_x[row, col] = is_edge_x
        self.terrain_edges_y[row, col] = is_edge_y
        self.terrain_edges_c[row, col] = is_edge_c

        super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)

    def _get_terrain_mesh(
        self, difficulty: float, cfg: SubTerrainBaseCfg
    ) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cfg = cfg.copy()

        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed

        sub_terrain_hash = dict_to_md5_hash(cfg.to_dict())

        sub_terrain_cache_dir = os.path.join(self.cfg.cache_dir, sub_terrain_hash)
        sub_terrain_obj_filename = os.path.join(sub_terrain_cache_dir, "mesh.obj")
        sub_terrain_csv_filename = os.path.join(sub_terrain_cache_dir, "origin.csv")
        sub_terrain_meta_filename = os.path.join(sub_terrain_cache_dir, "cfg.yaml")
        sub_terrain_edges_x_filename = os.path.join(sub_terrain_cache_dir, "edges_x.npy")
        sub_terrain_edges_y_filename = os.path.join(sub_terrain_cache_dir, "edges_y.npy")
        sub_terrain_edges_corners_filename = os.path.join(sub_terrain_cache_dir, "edges_c.npy")

        if all(
            [
                self.cfg.use_cache,
                os.path.exists(sub_terrain_obj_filename),
                os.path.exists(sub_terrain_csv_filename),
                os.path.exists(sub_terrain_edges_x_filename),
                os.path.exists(sub_terrain_edges_y_filename),
                os.path.exists(sub_terrain_edges_corners_filename),
            ]
        ):
            mesh = trimesh.load_mesh(sub_terrain_obj_filename, process=False)
            origin = np.loadtxt(sub_terrain_csv_filename, delimiter=",")

            is_edge_x = np.load(sub_terrain_edges_x_filename)
            is_edge_y = np.load(sub_terrain_edges_y_filename)
            is_edge_c = np.load(sub_terrain_edges_corners_filename)

            return mesh, origin, is_edge_x, is_edge_y, is_edge_c

        meshes, origin, is_edge_x, is_edge_y, is_edge_c = cfg.function(difficulty, cfg)
        mesh = trimesh.util.concatenate(meshes)

        transform = np.eye(4)
        transform[0:2, -1] = -cfg.size[0] * 0.5, -cfg.size[1] * 0.5
        mesh.apply_transform(transform)

        origin += transform[0:3, -1]

        if self.cfg.use_cache:
            os.makedirs(sub_terrain_cache_dir, exist_ok=True)

            mesh.export(sub_terrain_obj_filename)
            np.savetxt(sub_terrain_csv_filename, origin, delimiter=",", header="x,y,z")
            dump_yaml(sub_terrain_meta_filename, cfg)

            np.save(sub_terrain_edges_x_filename, is_edge_x)
            np.save(sub_terrain_edges_y_filename, is_edge_y)
            np.save(sub_terrain_edges_corners_filename, is_edge_c)

        return mesh, origin, is_edge_x, is_edge_y, is_edge_c
