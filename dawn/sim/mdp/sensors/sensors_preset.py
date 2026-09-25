import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

from dawn.sim.mdp.sensors.tiled_camera_cfg import TiledCameraCfg
from dawn.sim.utils.math_utils import quat_from_euler_xyz_tuple_float


def get_height_scanner_cfg() -> RayCasterCfg:
    return RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


def get_forward_height_map_cfg() -> RayCasterCfg:
    return RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(1.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2.0, 2.4]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


def get_contact_forces_cfg() -> ContactSensorCfg:
    return ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)


def _generate_env_id_regex(max_env_id: int) -> str:
    if max_env_id < 0:
        raise ValueError(f"max_env_id must be non-negative, got {max_env_id}")
    if max_env_id == 0:
        return "0"

    patterns = []
    max_str = str(max_env_id)
    num_digits = len(max_str)

    if max_env_id >= 9:
        patterns.append("[0-9]")
    else:
        patterns.append(f"[0-{max_env_id}]")

    if num_digits >= 2:
        if max_env_id >= 99:
            patterns.append("[1-9][0-9]")
        elif max_env_id >= 10:
            max_tens = max_env_id // 10
            max_ones = max_env_id % 10
            if max_tens == 1:
                patterns.append(f"1[0-{max_ones}]")
            else:
                patterns.append(f"[1-{max_tens - 1}][0-9]")
                patterns.append(f"{max_tens}[0-{max_ones}]")

    if num_digits >= 3:
        if max_env_id >= 999:
            patterns.append("[1-9][0-9][0-9]")
        elif max_env_id >= 100:
            max_hundreds = max_env_id // 100
            remainder = max_env_id % 100
            if max_hundreds == 1:
                if remainder >= 99:
                    patterns.append("1[0-9][0-9]")
                else:
                    max_tens = remainder // 10
                    max_ones = remainder % 10
                    if max_tens == 0:
                        patterns.append(f"10[0-{max_ones}]")
                    else:
                        patterns.append(f"1[0-{max_tens - 1}][0-9]")
                        patterns.append(f"1{max_tens}[0-{max_ones}]")
            else:
                patterns.append(f"[1-{max_hundreds - 1}][0-9][0-9]")
                if remainder >= 99:
                    patterns.append(f"{max_hundreds}[0-9][0-9]")
                else:
                    max_tens = remainder // 10
                    max_ones = remainder % 10
                    if max_tens == 0:
                        patterns.append(f"{max_hundreds}0[0-{max_ones}]")
                    else:
                        patterns.append(f"{max_hundreds}[0-{max_tens - 1}][0-9]")
                        patterns.append(f"{max_hundreds}{max_tens}[0-{max_ones}]")

    if num_digits >= 4:
        max_thousands = max_env_id // 1000
        remainder = max_env_id % 1000
        if max_thousands == 1:
            if remainder >= 999:
                patterns.append("1[0-9][0-9][0-9]")
            else:
                max_hundreds = remainder // 100
                remainder2 = remainder % 100
                if max_hundreds == 0:
                    if remainder2 >= 99:
                        patterns.append("10[0-9][0-9]")
                    else:
                        max_tens = remainder2 // 10
                        max_ones = remainder2 % 10
                        if max_tens == 0:
                            patterns.append(f"100[0-{max_ones}]")
                        else:
                            patterns.append(f"10[0-{max_tens - 1}][0-9]")
                            patterns.append(f"10{max_tens}[0-{max_ones}]")
                else:
                    patterns.append("10[0-9][0-9]")
                    if remainder2 >= 99:
                        patterns.append(f"1{max_hundreds}[0-9][0-9]")
                    else:
                        max_tens = remainder2 // 10
                        max_ones = remainder2 % 10
                        if max_tens == 0:
                            patterns.append(f"1{max_hundreds}0[0-{max_ones}]")
                        else:
                            patterns.append(f"1{max_hundreds}[0-{max_tens - 1}][0-9]")
                            patterns.append(f"1{max_hundreds}{max_tens}[0-{max_ones}]")
        else:
            patterns.append(f"[1-{max_thousands - 1}][0-9][0-9][0-9]")
            if remainder >= 999:
                patterns.append(f"{max_thousands}[0-9][0-9][0-9]")
            else:
                max_hundreds = remainder // 100
                remainder2 = remainder % 100
                if max_hundreds == 0:
                    if remainder2 >= 99:
                        patterns.append(f"{max_thousands}0[0-9][0-9]")
                    else:
                        max_tens = remainder2 // 10
                        max_ones = remainder2 % 10
                        if max_tens == 0:
                            patterns.append(f"{max_thousands}00[0-{max_ones}]")
                        else:
                            patterns.append(f"{max_thousands}0[0-{max_tens - 1}][0-9]")
                            patterns.append(f"{max_thousands}0{max_tens}[0-{max_ones}]")
                else:
                    patterns.append(f"{max_thousands}[0-{max_hundreds - 1}][0-9][0-9]")
                    if remainder2 >= 99:
                        patterns.append(f"{max_thousands}{max_hundreds}[0-9][0-9]")
                    else:
                        max_tens = remainder2 // 10
                        max_ones = remainder2 % 10
                        if max_tens == 0:
                            patterns.append(f"{max_thousands}{max_hundreds}0[0-{max_ones}]")
                        else:
                            patterns.append(f"{max_thousands}{max_hundreds}[0-{max_tens - 1}][0-9]")
                            patterns.append(f"{max_thousands}{max_hundreds}{max_tens}[0-{max_ones}]")

    return "(" + "|".join(patterns) + ")"


def get_depth_camera_cfg(camera_num_envs: int | None = None, height: int = 64, width: int = 64) -> TiledCameraCfg:
    if height <= 0:
        raise ValueError(f"height must be positive, got {height}")
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")

    if camera_num_envs is None:
        env_id_regex = "[0-9]+"
    else:
        max_env_id = camera_num_envs - 1
        env_id_regex = _generate_env_id_regex(max_env_id)

    prim_path = f"/World/envs/env_{env_id_regex}/Robot/trunk/custom_cam"

    return TiledCameraCfg(
        prim_path=prim_path,
        update_period=0.1,
        height=height,
        width=width,
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.28, 5.0)
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.30, 0.0, 0.12), rot=quat_from_euler_xyz_tuple_float(0.0, 0.0, 0.0), convention="world"
        ),
    )
