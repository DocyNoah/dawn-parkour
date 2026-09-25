from dawn.config.experiment_cfg import AppCfg, EnvCfg, EnvWrapperCfg, ExperimentCfg, LoggingCfg
from dawn.rl.configs.train_cfg import ActorCriticPolicyCfg, DawnTrainCfg, TrainRunnerCfg
from dawn.rl.configs.world_model_cfg import DAWNWorldModelCfg


def make_dawn_experiment_config() -> ExperimentCfg:
    return ExperimentCfg(
        app=AppCfg(enable_cameras=True),
        env=EnvCfg(
            task="Go1-DAWN-v0",
            invisible_robot=True,
            num_envs=4096,
            camera_num_envs=1024,
        ),
        wrappers=EnvWrapperCfg(
            use_depth_buffer_wrapper=True,
            use_depth_noise_wrapper=True,
            use_visibility_wrapper=True,
            use_camera_follow_wrapper=True,
            use_visualization_wrapper=True,
        ),
        train=DawnTrainCfg(
            algorithm_name="dawn",
            runner=TrainRunnerCfg(max_iterations=1_000_000),
            policy=ActorCriticPolicyCfg(
                actor_network_name="WMActorNetwork",
                critic_network_name="WMCriticNetwork",
            ),
            wm=DAWNWorldModelCfg(),
        ),
        logging=LoggingCfg(
            experiment_name="DAWN",
            run_name="DAWN",
        ),
    )


__all__ = ["make_dawn_experiment_config"]
