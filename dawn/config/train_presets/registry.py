from dawn.config.experiment_cfg import ExperimentCfg
from dawn.config.train_presets.dawn import make_dawn_experiment_config


def get_train_presets() -> dict[str, tuple[str, ExperimentCfg]]:
    return {"dawn": ("DAWN", make_dawn_experiment_config())}
