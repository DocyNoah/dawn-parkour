from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from dawn.sim.envs.go1_root import Go1RootEnvCfg, MySceneCfg
from dawn.sim.mdp.commands.commands_presets.vision_commands import VisionCommandsPreset
from dawn.sim.mdp.curriculums import curriculums
from dawn.sim.mdp.sensors.sensors_preset import get_depth_camera_cfg
from dawn.sim.mdp.terminations import terminations
from dawn.sim.mdp.terrains.go1_parkour_terrain import terrain_generator_cfg
from dawn.sim.mdp.terrains.go1_parkour_terrain_play import terrain_generator_cfg as play_terrain_generator_cfg
from dawn.sim.mdp.terrains.terrain_importer import TerrainImporterWithEdges


@configclass
class DepthCfg:
    use_camera: bool = True
    camera_num_envs: int | None = None
    original_image_shape: tuple[int, int] = (64, 64)
    resized_image_shape: tuple[int, int] = (64, 64)

    update_interval: int = 5

    near_clip: float = 0.28
    far_clip: float = 2.0
    scale: float = 1.0
    buffer_len: int = 2


@configclass
class ObsCfg:
    prop_dim: int = 33
    priv_dim: int = 54
    action_dim: int = 12
    heightmap_dim: int = 187
    forward_height_dim: int = 525


@configclass
class Go1DawnRootEnvCfg(Go1RootEnvCfg):
    commands: VisionCommandsPreset = VisionCommandsPreset()
    scene: MySceneCfg = MySceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=False,
    )
    depth: DepthCfg = DepthCfg(camera_num_envs=1024)
    obs_num: ObsCfg = ObsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 20.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.terminations.fall = DoneTerm(
            func=terminations.root_height_below_minimum,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
                "minimum_height": 0.4,
                "use_terrain_type": True,
                "terrain_type_names_list": ["gap"],
            },
        )

        self.terminations.max_level_clear = DoneTerm(
            func=terminations.is_max_level_clear,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="trunk")},
            time_out=True,
        )

        self.terminations.off_center = DoneTerm(
            func=terminations.off_center_termination,
            params={"max_distance": 0.4},
        )

        self.terminations.no_progress = DoneTerm(
            func=terminations.no_progress,
            params={"max_stagnation_steps": 100, "command_name": "base_velocity", "min_command_speed": 0.02},
        )

        self.rewards.edge_avoidance.params["use_terrain_type"] = True
        self.rewards.edge_avoidance.params["terrain_type_names_list"] = ["gap", "step"]
        self.rewards.stumble.params["use_terrain_type"] = True
        self.rewards.stumble.params["terrain_type_names_list"] = ["gap", "step"]

        self.scene.terrain.terrain_generator = terrain_generator_cfg
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            self.scene.terrain.terrain_generator.curriculum = True

        self.events.reset_base.params["pose_range"]["x"] = (-0.05, 0.05)
        self.events.reset_base.params["pose_range"]["y"] = (-0.05, 0.05)
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["roll"] = (-0.0, 0.0)
        self.events.reset_base.params["pose_range"]["pitch"] = (-0.0, 0.0)

        curriculum = self.scene.terrain.terrain_generator.curriculum
        self.scene.terrain.class_type = TerrainImporterWithEdges
        self.scene.terrain.edge_width_thresh = 0.05
        self.scene.terrain.terrain_generator = terrain_generator_cfg
        self.scene.terrain.terrain_generator.curriculum = curriculum

        self.scene.terrain.debug_vis = False

        self.curriculum.terrain_levels.func = curriculums.terrain_levels_vel_stats

        self.sim.physx.gpu_collision_stack_size = 2**27

        if self.depth.camera_num_envs > self.scene.num_envs:
            raise ValueError(
                f"depth.camera_num_envs ({self.depth.camera_num_envs}) "
                f"must be <= scene.num_envs ({self.scene.num_envs})"
            )
        self.scene.depth_camera = get_depth_camera_cfg(
            camera_num_envs=self.depth.camera_num_envs,
            height=self.depth.original_image_shape[0],
            width=self.depth.original_image_shape[1],
        )


def set_play_config(env_cfg: Go1DawnRootEnvCfg, remove_pushing: bool = False) -> None:
    env_cfg.scene.num_envs = 4
    env_cfg.depth.camera_num_envs = env_cfg.scene.num_envs
    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.terrain.terrain_generator = play_terrain_generator_cfg
    env_cfg.scene.terrain.terrain_generator.curriculum = True
    env_cfg.scene.terrain.max_init_terrain_level = None
    env_cfg.curriculum.terrain_levels.func = curriculums.terrain_levels_vel_stats
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0

    env_cfg.events.reset_base.params["pose_range"]["x"] = (-0.02, 0.02)
    env_cfg.events.reset_base.params["pose_range"]["y"] = (-0.02, 0.02)
    env_cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
    env_cfg.events.reset_base.params["pose_range"]["roll"] = (0.0, 0.0)
    env_cfg.events.reset_base.params["pose_range"]["pitch"] = (0.0, 0.0)

    if remove_pushing:
        env_cfg.events.base_external_force_torque = None
        env_cfg.events.push_robot = None
