# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

import carb  # type: ignore
import numpy as np
import torch
import warp as wp
from isaaclab.sensors.camera import Camera
from isaaclab.sensors.sensor_base import SensorBase
from isaaclab.utils.warp.kernels import reshape_tiled_image
from isaacsim.core.prims import XFormPrim  # type: ignore
from isaacsim.core.version import get_version  # type: ignore
from pxr import UsdGeom  # type: ignore

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .tiled_camera_cfg import TiledCameraCfg


class TiledCamera(Camera):
    cfg: TiledCameraCfg

    def __init__(self, cfg: TiledCameraCfg):
        isaac_sim_version = float(".".join(get_version()[2:4]))
        if isaac_sim_version < 4.2:
            raise RuntimeError(
                f"TiledCamera is only available from Isaac Sim 4.2.0. Current version is {isaac_sim_version}. Please"
                " update to Isaac Sim 4.2.0"
            )
        super().__init__(cfg)

    def __del__(self):
        SensorBase.__del__(self)

        for annotator in self._annotators.values():
            annotator.detach(self.render_product_paths)

    def __str__(self) -> str:
        return (
            f"Tiled Camera @ '{self.cfg.prim_path}': \n"
            f"\tdata types   : {list(self.data.output.keys())} \n"
            f"\tsemantic filter : {self.cfg.semantic_filter}\n"
            f"\tcolorize semantic segm.   : {self.cfg.colorize_semantic_segmentation}\n"
            f"\tcolorize instance segm.   : {self.cfg.colorize_instance_segmentation}\n"
            f"\tcolorize instance id segm.: {self.cfg.colorize_instance_id_segmentation}\n"
            f"\tupdate period (s): {self.cfg.update_period}\n"
            f"\tshape        : {self.image_shape}\n"
            f"\tnumber of sensors : {self._view.count}"
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if not self._is_initialized:
            raise RuntimeError(
                "TiledCamera could not be initialized. Please ensure --enable_cameras is used to enable rendering."
            )

        max_env_id = self._view.count - 1
        if env_ids is not None:
            if isinstance(env_ids, (list, tuple)):
                env_ids = [eid for eid in env_ids if 0 <= eid <= max_env_id]
            elif isinstance(env_ids, torch.Tensor):
                env_ids = [eid.item() for eid in env_ids if 0 <= eid.item() <= max_env_id]
            elif isinstance(env_ids, slice):
                if env_ids == slice(None):
                    env_ids = list(range(self._view.count))
                else:
                    env_ids = []
            else:
                env_ids = [env_ids] if 0 <= env_ids <= max_env_id else []

        if not env_ids:
            return

        SensorBase.reset(self, env_ids)

        if env_ids is None:
            env_ids = slice(None)

        self._frame[env_ids] = 0

    def _initialize_impl(self) -> None:
        carb_settings_iface = carb.settings.get_settings()
        if not carb_settings_iface.get("/isaaclab/cameras_enabled"):
            raise RuntimeError(
                "A camera was spawned without the --enable_cameras flag. Please use --enable_cameras to enable"
                " rendering."
            )

        import omni.replicator.core as rep  # type: ignore

        SensorBase._initialize_impl(self)  # noqa: SLF001

        self._view = XFormPrim(self.cfg.prim_path, reset_xform_properties=False)
        self._view.initialize()

        if self._view.count != self._num_envs:
            raise RuntimeError(
                f"Number of camera prims in the view ({self._view.count}) does not match"
                f" the number of environments ({self._num_envs})."
            )

        self._ALL_INDICES = torch.arange(self._view.count, device=self._device, dtype=torch.long)

        self._frame = torch.zeros(self._view.count, device=self._device, dtype=torch.long)

        for cam_prim_path in self._view.prim_paths:
            cam_prim = self.stage.GetPrimAtPath(cam_prim_path)

            if not cam_prim.IsA(UsdGeom.Camera):
                raise RuntimeError(f"Prim at path '{cam_prim_path}' is not a Camera.")

            sensor_prim = UsdGeom.Camera(cam_prim)
            self._sensor_prims.append(sensor_prim)

        rp = rep.create.render_product_tiled(
            cameras=self._view.prim_paths, tile_resolution=(self.cfg.width, self.cfg.height)
        )
        self._render_product_paths = [rp.path]

        self._annotators = {}
        for annotator_type in self.cfg.data_types:
            if annotator_type == "rgba" or annotator_type == "rgb":
                annotator = rep.AnnotatorRegistry.get_annotator("rgb", device=self.device, do_array_copy=False)
                self._annotators["rgba"] = annotator
            elif annotator_type == "depth" or annotator_type == "distance_to_image_plane":
                annotator = rep.AnnotatorRegistry.get_annotator(
                    "distance_to_image_plane", device=self.device, do_array_copy=False
                )
                self._annotators[annotator_type] = annotator

            else:
                init_params = None
                if annotator_type == "semantic_segmentation":
                    init_params = {
                        "colorize": self.cfg.colorize_semantic_segmentation,
                        "mapping": json.dumps(self.cfg.semantic_segmentation_mapping),
                    }
                elif annotator_type == "instance_segmentation_fast":
                    init_params = {"colorize": self.cfg.colorize_instance_segmentation}
                elif annotator_type == "instance_id_segmentation_fast":
                    init_params = {"colorize": self.cfg.colorize_instance_id_segmentation}

                annotator = rep.AnnotatorRegistry.get_annotator(
                    annotator_type, init_params, device=self.device, do_array_copy=False
                )
                self._annotators[annotator_type] = annotator

        for annotator in self._annotators.values():
            annotator.attach(self._render_product_paths)

        self._create_buffers()

    def _update_buffers_impl(self, env_ids: Sequence[int]) -> None:
        self._frame[env_ids] += 1

        if self.cfg.update_latest_camera_pose:
            self._update_poses(env_ids)

        for data_type, annotator in self._annotators.items():
            output = annotator.get_data()
            if isinstance(output, dict):
                tiled_data_buffer = output["data"]
                self._data.info[data_type] = output["info"]
            else:
                tiled_data_buffer = output

            if isinstance(tiled_data_buffer, np.ndarray):
                tiled_data_buffer = wp.array(tiled_data_buffer, device=self.device)
            else:
                tiled_data_buffer = tiled_data_buffer.to(device=self.device)

            if (
                (data_type == "semantic_segmentation" and self.cfg.colorize_semantic_segmentation)
                or (data_type == "instance_segmentation_fast" and self.cfg.colorize_instance_segmentation)
                or (data_type == "instance_id_segmentation_fast" and self.cfg.colorize_instance_id_segmentation)
            ):
                tiled_data_buffer = wp.array(
                    ptr=tiled_data_buffer.ptr, shape=(*tiled_data_buffer.shape, 4), dtype=wp.uint8, device=self.device
                )

            if data_type == "motion_vectors":
                tiled_data_buffer = tiled_data_buffer[:, :, :2].contiguous()

            wp.launch(
                kernel=reshape_tiled_image,
                dim=(self._view.count, self.cfg.height, self.cfg.width),
                inputs=[
                    tiled_data_buffer.flatten(),
                    wp.from_torch(self._data.output[data_type]),
                    *list(self._data.output[data_type].shape[1:]),
                    self._tiling_grid_shape()[0],
                ],
                device=self.device,
            )

            if data_type == "rgba" and "rgb" in self.cfg.data_types:
                self._data.output["rgb"] = self._data.output["rgba"][..., :3]

            if data_type == "distance_to_camera":
                self._data.output[data_type][self._data.output[data_type] > self.cfg.spawn.clipping_range[1]] = (
                    torch.inf
                )

            if (
                data_type == "distance_to_camera" or data_type == "distance_to_image_plane" or data_type == "depth"
            ) and self.cfg.depth_clipping_behavior != "none":
                self._data.output[data_type][torch.isinf(self._data.output[data_type])] = (
                    0.0 if self.cfg.depth_clipping_behavior == "zero" else self.cfg.spawn.clipping_range[1]
                )

    def _check_supported_data_types(self, cfg: TiledCameraCfg) -> None:
        common_elements = set(cfg.data_types) & Camera.UNSUPPORTED_TYPES
        if common_elements:
            fast_common_elements = []
            for item in common_elements:
                if "instance_segmentation" in item or "instance_id_segmentation" in item:
                    fast_common_elements.append(item + "_fast")

            raise ValueError(
                f"TiledCamera class does not support the following sensor types: {common_elements}."
                "\n\tThis is because these sensor types output numpy structured data types which"
                "can't be converted to torch tensors easily."
                "\n\tHint: If you need to work with these sensor types, we recommend using their fast counterparts."
                f"\n\t\tFast counterparts: {fast_common_elements}"
            )

    def _create_buffers(self) -> None:
        self._data.pos_w = torch.zeros((self._view.count, 3), device=self._device)
        self._data.quat_w_world = torch.zeros((self._view.count, 4), device=self._device)
        self._update_poses(self._ALL_INDICES)

        self._data.intrinsic_matrices = torch.zeros((self._view.count, 3, 3), device=self._device)
        self._update_intrinsic_matrices(self._ALL_INDICES)
        self._data.image_shape = self.image_shape

        data_dict = {}
        if "rgba" in self.cfg.data_types or "rgb" in self.cfg.data_types:
            data_dict["rgba"] = torch.zeros(
                (self._view.count, self.cfg.height, self.cfg.width, 4), device=self.device, dtype=torch.uint8
            ).contiguous()
        if "rgb" in self.cfg.data_types:
            data_dict["rgb"] = data_dict["rgba"][..., :3]
        if "distance_to_image_plane" in self.cfg.data_types:
            data_dict["distance_to_image_plane"] = torch.zeros(
                (self._view.count, self.cfg.height, self.cfg.width, 1), device=self.device, dtype=torch.float32
            ).contiguous()
        if "depth" in self.cfg.data_types:
            data_dict["depth"] = torch.zeros(
                (self._view.count, self.cfg.height, self.cfg.width, 1), device=self.device, dtype=torch.float32
            ).contiguous()
        if "distance_to_camera" in self.cfg.data_types:
            data_dict["distance_to_camera"] = torch.zeros(
                (self._view.count, self.cfg.height, self.cfg.width, 1), device=self.device, dtype=torch.float32
            ).contiguous()
        if "normals" in self.cfg.data_types:
            data_dict["normals"] = torch.zeros(
                (self._view.count, self.cfg.height, self.cfg.width, 3), device=self.device, dtype=torch.float32
            ).contiguous()
        if "motion_vectors" in self.cfg.data_types:
            data_dict["motion_vectors"] = torch.zeros(
                (self._view.count, self.cfg.height, self.cfg.width, 2), device=self.device, dtype=torch.float32
            ).contiguous()
        if "semantic_segmentation" in self.cfg.data_types:
            if self.cfg.colorize_semantic_segmentation:
                data_dict["semantic_segmentation"] = torch.zeros(
                    (self._view.count, self.cfg.height, self.cfg.width, 4), device=self.device, dtype=torch.uint8
                ).contiguous()
            else:
                data_dict["semantic_segmentation"] = torch.zeros(
                    (self._view.count, self.cfg.height, self.cfg.width, 1), device=self.device, dtype=torch.int32
                ).contiguous()
        if "instance_segmentation_fast" in self.cfg.data_types:
            if self.cfg.colorize_instance_segmentation:
                data_dict["instance_segmentation_fast"] = torch.zeros(
                    (self._view.count, self.cfg.height, self.cfg.width, 4), device=self.device, dtype=torch.uint8
                ).contiguous()
            else:
                data_dict["instance_segmentation_fast"] = torch.zeros(
                    (self._view.count, self.cfg.height, self.cfg.width, 1), device=self.device, dtype=torch.int32
                ).contiguous()
        if "instance_id_segmentation_fast" in self.cfg.data_types:
            if self.cfg.colorize_instance_id_segmentation:
                data_dict["instance_id_segmentation_fast"] = torch.zeros(
                    (self._view.count, self.cfg.height, self.cfg.width, 4), device=self.device, dtype=torch.uint8
                ).contiguous()
            else:
                data_dict["instance_id_segmentation_fast"] = torch.zeros(
                    (self._view.count, self.cfg.height, self.cfg.width, 1), device=self.device, dtype=torch.int32
                ).contiguous()

        self._data.output = data_dict
        self._data.info = {}

    def _tiled_image_shape(self) -> tuple[int, int]:
        cols, rows = self._tiling_grid_shape()
        return (self.cfg.width * cols, self.cfg.height * rows)

    def _tiling_grid_shape(self) -> tuple[int, int]:
        cols = math.ceil(math.sqrt(self._view.count))
        rows = math.ceil(self._view.count / cols)
        return (cols, rows)

    def _create_annotator_data(self) -> None:
        raise RuntimeError("This function should not be called for the tiled camera sensor.")

    def _process_annotator_output(self, name: str, output: Any) -> tuple[torch.tensor, dict | None]:
        raise RuntimeError("This function should not be called for the tiled camera sensor.")

    def _invalidate_initialize_callback(self, event: Any) -> None:
        super()._invalidate_initialize_callback(event)

        self._view = None
