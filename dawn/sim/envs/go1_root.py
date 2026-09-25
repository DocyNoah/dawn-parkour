from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from dawn.sim.mdp.actions.actions_preset import ActionsPreset
from dawn.sim.mdp.assets.robot_preset import get_robot_cfg
from dawn.sim.mdp.commands.commands_presets.blind_commands import BlindCommandsPreset
from dawn.sim.mdp.curriculums import terrain_levels_vel_only_up
from dawn.sim.mdp.curriculums.curriculums_preset import CurriculumsPreset
from dawn.sim.mdp.events.events_preset import EventsPreset
from dawn.sim.mdp.rewards.rewards_preset import RewardsPreset
from dawn.sim.mdp.scene_entities.lighting_preset import get_sky_light_cfg
from dawn.sim.mdp.sensors.sensors_preset import (
    get_contact_forces_cfg,
    get_depth_camera_cfg,
    get_forward_height_map_cfg,
    get_height_scanner_cfg,
)
from dawn.sim.mdp.terminations.terminations_preset import TerminationsPreset
from dawn.sim.mdp.terrains.go1_parkour_terrain import terrain_generator_cfg
from dawn.sim.mdp.terrains.terrain_importer_preset import get_terrain_cfg


@configclass
class MySceneCfg(InteractiveSceneCfg):
    terrain = get_terrain_cfg(terrain_generator_cfg)

    robot: ArticulationCfg = get_robot_cfg()

    contact_forces = get_contact_forces_cfg()
    height_scanner = get_height_scanner_cfg()
    forward_height_scanner = get_forward_height_map_cfg()
    depth_camera = get_depth_camera_cfg()

    sky_light = get_sky_light_cfg()

    def __post_init__(self):
        super().__post_init__()

        self.robot.actuators["base_legs"].stiffness = 40.0
        self.robot.actuators["base_legs"].damping = 1.0


@configclass
class Go1RootEnvCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)

    actions: ActionsPreset = ActionsPreset()
    commands: BlindCommandsPreset = BlindCommandsPreset()

    rewards: RewardsPreset = RewardsPreset()
    terminations: TerminationsPreset = TerminationsPreset()
    events: EventsPreset = EventsPreset()
    curriculum: CurriculumsPreset = CurriculumsPreset()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.enable_cameras = True

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


def _remove_pushing_events(env_cfg: Go1RootEnvCfg, remove_pushing: bool) -> None:
    if remove_pushing:
        env_cfg.events.base_external_force_torque = None
        env_cfg.events.push_robot = None


def set_play_config(env_cfg: Go1RootEnvCfg, remove_pushing: bool = False) -> None:
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.terrain.max_init_terrain_level = None

    if env_cfg.curriculum.terrain_levels is not None:
        env_cfg.curriculum.terrain_levels.func = terrain_levels_vel_only_up
    _remove_pushing_events(env_cfg, remove_pushing)
