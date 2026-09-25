from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.terrains.terrain_importer import TerrainImporter
from scipy.ndimage import binary_dilation

from dawn.sim.mdp.terrains.terrain_generator import TerrainGeneratorWithEdges

if TYPE_CHECKING:
    import trimesh
    import warp

    from dawn.sim.mdp.terrains.terrain_importer_cfg import TerrainImporterWithEdgesCfg


EDGE_X_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "edge_x": sim_utils.CylinderCfg(
            radius=0.005,
            height=0.1,
            axis="Y",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)


EDGE_Y_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "edge_y": sim_utils.CylinderCfg(
            radius=0.005,
            height=0.1,
            axis="X",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    }
)


EDGE_C_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "edge_c": sim_utils.CuboidCfg(
            size=(0.02, 0.02, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    }
)


class TerrainImporterWithEdges(TerrainImporter):
    meshes: dict[str, trimesh.Trimesh]

    warp_meshes: dict[str, warp.Mesh]

    terrain_origins: torch.Tensor | None

    env_origins: torch.Tensor

    _terrain_edges_x: np.ndarray | None

    _terrain_edges_y: np.ndarray | None

    _terrain_edges_c: np.ndarray | None

    _global_terrain_edges_x: np.ndarray | None

    _global_terrain_edges_y: np.ndarray | None

    _global_terrain_edges_c: np.ndarray | None

    def __init__(self, cfg: TerrainImporterWithEdgesCfg):
        cfg.validate()

        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore

        self.terrain_prim_paths = []
        self.env_origins = None
        self.terrain_origins = None
        self.terrain_generator = None

        self._terrain_flat_patches = {}

        self._terrain_edges_x = None
        self._terrain_edges_y = None
        self._terrain_edges_c = None
        self._global_terrain_edges_x = None
        self._global_terrain_edges_y = None
        self._global_terrain_edges_c = None

        if self.cfg.terrain_type == "generator":
            if self.cfg.terrain_generator is None:
                raise ValueError("Input terrain type is 'generator' but no value provided for 'terrain_generator'.")

            terrain_generator = TerrainGeneratorWithEdges(cfg=self.cfg.terrain_generator, device=self.device)
            self.import_mesh("terrain", terrain_generator.terrain_mesh)

            self.configure_env_origins(terrain_generator.terrain_origins)

            self._terrain_flat_patches = terrain_generator.flat_patches

            self._terrain_edges_x = terrain_generator.terrain_edges_x.copy()
            self._terrain_edges_y = terrain_generator.terrain_edges_y.copy()
            self._terrain_edges_c = terrain_generator.terrain_edges_c.copy()

            half_edge_width = int(self.cfg.edge_width_thresh / self.cfg.terrain_generator.horizontal_scale)
            self._global_terrain_edges_x = self._compute_global_edge_map(
                terrain_generator.global_terrain_edges_x, "x", half_edge_width
            )
            self._global_terrain_edges_y = self._compute_global_edge_map(
                terrain_generator.global_terrain_edges_y, "y", half_edge_width
            )
            self._global_terrain_edges_c = self._compute_global_edge_map(
                terrain_generator.global_terrain_edges_c, "c", half_edge_width
            )

            self.is_flat = torch.tensor(
                terrain_generator.is_flat_terrain[self.terrain_types.cpu().numpy()],
                dtype=torch.bool,
                device=self.device,
            )

            self.terrain_type_names = terrain_generator.terrain_type_names

        elif self.cfg.terrain_type == "usd":
            if self.cfg.usd_path is None:
                raise ValueError("Input terrain type is 'usd' but no value provided for 'usd_path'.")

            self.import_usd("terrain", self.cfg.usd_path)

            self.configure_env_origins()

            self.is_flat = None
            self.terrain_type_names = None
        elif self.cfg.terrain_type == "plane":
            self.import_ground_plane("terrain")

            self.configure_env_origins()

            self.is_flat = None
            self.terrain_type_names = None
        else:
            raise ValueError(f"Terrain type '{self.cfg.terrain_type}' not available.")

        self.set_debug_vis(self.cfg.debug_vis)

    @property
    def has_edge_info(self) -> bool:
        return self._terrain_edges_x is not None

    @property
    def terrain_edges_x(self) -> np.ndarray | None:
        return self._terrain_edges_x.copy() if self._terrain_edges_x is not None else None

    @property
    def terrain_edges_y(self) -> np.ndarray | None:
        return self._terrain_edges_y.copy() if self._terrain_edges_y is not None else None

    @property
    def terrain_edges_c(self) -> np.ndarray | None:
        return self._terrain_edges_c.copy() if self._terrain_edges_c is not None else None

    @property
    def global_terrain_edges_x(self) -> np.ndarray | None:
        return self._global_terrain_edges_x.copy() if self._global_terrain_edges_x is not None else None

    @property
    def global_terrain_edges_y(self) -> np.ndarray | None:
        return self._global_terrain_edges_y.copy() if self._global_terrain_edges_y is not None else None

    @property
    def global_terrain_edges_c(self) -> np.ndarray | None:
        return self._global_terrain_edges_c.copy() if self._global_terrain_edges_c is not None else None

    def get_edge_info(self, row: int, col: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if self._terrain_edges_x is not None:
            return (
                self._terrain_edges_x[row, col].copy(),
                self._terrain_edges_y[row, col].copy(),
                self._terrain_edges_c[row, col].copy(),
            )
        return None

    def get_edge_count(self, row: int, col: int) -> dict[str, int]:
        if self._terrain_edges_x is not None:
            is_edge_x = self._terrain_edges_x[row, col]
            is_edge_y = self._terrain_edges_y[row, col]
            is_edge_c = self._terrain_edges_c[row, col]
            return {
                "x_edges": int(np.sum(is_edge_x)),
                "y_edges": int(np.sum(is_edge_y)),
                "corner_edges": int(np.sum(is_edge_c)),
                "total_edges": int(np.sum(is_edge_x) + np.sum(is_edge_y) + np.sum(is_edge_c)),
            }
        return {"x_edges": 0, "y_edges": 0, "corner_edges": 0, "total_edges": 0}

    def get_all_edge_statistics(self) -> dict[str, float]:
        if self._terrain_edges_x is not None:
            num_rows, num_cols = self._terrain_edges_x.shape[:2]
            edge_counts = []
            terrains_with_edges = 0

            for row in range(num_rows):
                for col in range(num_cols):
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
        return {
            "mean_edges": 0.0,
            "max_edges": 0,
            "min_edges": 0,
            "total_terrains_with_edges": 0,
            "percentage_with_edges": 0.0,
        }

    def __str__(self) -> str:
        msg = "Terrain Importer:"
        msg += f"\n\tTerrain type: {self.cfg.terrain_type}"
        msg += f"\n\tNumber of environments: {self.cfg.num_envs}"
        msg += f"\n\tEnvironment origins shape: {self.env_origins.shape if self.env_origins is not None else 'None'}"

        if self.terrain_origins is not None:
            msg += f"\n\tTerrain origins shape: {self.terrain_origins.shape}"

        if self.has_edge_info:
            edge_stats = self.get_all_edge_statistics()
            if edge_stats["total_terrains_with_edges"] > 0:
                msg += "\n\tEdge information available: Yes"
                msg += f"\n\tTerrains with edges: {edge_stats['total_terrains_with_edges']}"
                msg += f"\n\tPercentage with edges: {edge_stats['percentage_with_edges']:.1f}%"
                msg += f"\n\tMean edges per terrain: {edge_stats['mean_edges']:.1f}"
                msg += f"\n\tMax edges in a terrain: {edge_stats['max_edges']}"
                msg += f"\n\tMin edges in a terrain: {edge_stats['min_edges']}"
            else:
                msg += "\n\tEdge information available: Yes (no edges detected)"
        else:
            msg += "\n\tEdge information available: No"

        return msg

    def set_debug_vis(self, debug_vis: bool) -> bool:
        parent_result = super().set_debug_vis(debug_vis)

        if debug_vis and self.has_edge_info:
            self._create_edge_visualizers()
            self._update_edge_visualizers()
            self._set_edge_visibility(True)
        else:
            self._set_edge_visibility(False)

        return parent_result

    def _create_edge_visualizers(self) -> None:
        if not hasattr(self, "edge_x_visualizer"):
            self.edge_x_visualizer = VisualizationMarkers(
                cfg=EDGE_X_MARKER_CFG.replace(prim_path="/Visuals/TerrainEdgesX")
            )

    def _update_edge_visualizers(self) -> None:
        if self._global_terrain_edges_x is not None:
            edge_x_data = self._get_edge_world_positions(self._global_terrain_edges_x)
            edge_y_data = self._get_edge_world_positions(self._global_terrain_edges_y)
            edge_c_data = self._get_edge_world_positions(self._global_terrain_edges_c)

            if hasattr(self, "edge_x_visualizer") and len(edge_x_data) > 0:
                self.edge_x_visualizer.visualize(edge_x_data)

            if hasattr(self, "edge_y_visualizer") and len(edge_y_data) > 0:
                self.edge_y_visualizer.visualize(edge_y_data)

            if hasattr(self, "edge_c_visualizer") and len(edge_c_data) > 0:
                self.edge_c_visualizer.visualize(edge_c_data)

    @staticmethod
    def _compute_global_edge_map(
        edge_map: np.ndarray,
        axis: Literal["x", "y", "c"],
        half_edge_width: int,
    ) -> np.ndarray:
        k = half_edge_width * 2 + 1
        if axis == "x":
            structure = np.ones((k, 1))
        elif axis == "y":
            structure = np.ones((1, k))
        else:
            structure = np.ones((k, k))
        return binary_dilation(edge_map, structure=structure)

    def _get_edge_world_positions(self, edge_map: np.ndarray, max_markers: int = 10000) -> torch.Tensor:
        if edge_map is None:
            return torch.empty((0, 3), device=self.device, dtype=torch.float32)

        edge_indices = np.where(edge_map)
        if len(edge_indices[0]) == 0:
            return torch.empty((0, 3), device=self.device, dtype=torch.float32)

        num_edges = len(edge_indices[0])
        if num_edges > max_markers:
            sample_indices = np.random.choice(num_edges, max_markers, replace=False)
            edge_x_indices = edge_indices[0][sample_indices]
            edge_y_indices = edge_indices[1][sample_indices]
        else:
            edge_x_indices = edge_indices[0]
            edge_y_indices = edge_indices[1]

        if hasattr(self.cfg, "terrain_generator") and self.cfg.terrain_generator is not None:
            horizontal_scale = self.cfg.terrain_generator.horizontal_scale
        else:
            horizontal_scale = 0.05

        world_x = edge_x_indices * horizontal_scale
        world_y = edge_y_indices * horizontal_scale

        if hasattr(self.cfg, "terrain_generator") and self.cfg.terrain_generator is not None:
            terrain_size = self.cfg.terrain_generator.size
            num_rows = self.cfg.terrain_generator.num_rows
            num_cols = self.cfg.terrain_generator.num_cols

            total_width = num_rows * terrain_size[0]
            total_height = num_cols * terrain_size[1]

            world_x = world_x - total_width / 2
            world_y = world_y - total_height / 2

        world_z = np.full_like(world_x, 0.1, dtype=np.float32)

        positions = np.stack([world_x, world_y, world_z], axis=1).astype(np.float32)

        return torch.from_numpy(positions).to(self.device)

    def _set_edge_visibility(self, visible: bool) -> None:
        if hasattr(self, "edge_x_visualizer"):
            self.edge_x_visualizer.set_visibility(visible)

        if hasattr(self, "edge_y_visualizer"):
            self.edge_y_visualizer.set_visibility(visible)

        if hasattr(self, "edge_c_visualizer"):
            self.edge_c_visualizer.set_visibility(visible)

    def _compute_env_origins_curriculum(self, num_envs: int, origins: torch.Tensor) -> torch.Tensor:
        num_rows, num_cols = origins.shape[:2]

        self.max_terrain_level = num_rows

        self.terrain_levels = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        self.terrain_types = (torch.arange(num_envs, device=self.device) % num_cols).to(torch.long)

        env_origins = torch.zeros(num_envs, 3, device=self.device)
        env_origins[:] = origins[self.terrain_levels, self.terrain_types]
        return env_origins
