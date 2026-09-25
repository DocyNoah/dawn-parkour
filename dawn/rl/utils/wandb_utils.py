from __future__ import annotations

import os

from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")


class WandbSummaryWriter(SummaryWriter):
    def __init__(self, log_dir: str, flush_secs: int, cfg: dict):
        super().__init__(log_dir, flush_secs)

        if cfg["run_name"]:
            run_name = cfg["run_name"] + "_" + cfg["now_time"]
        else:
            run_name = cfg["now_time"]

        wandb.init(
            project=cfg["wandb_project"],
            entity=os.environ.get("WANDB_USERNAME", None),
            group=cfg["experiment_name"],
            name=run_name,
            settings=wandb.Settings(start_method="fork"),
        )

        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

        run_name = os.path.split(log_dir)[-1]

        wandb.log({"log_dir": run_name})

    def store_config(self, env_cfg: dict, train_cfg: dict, logging_cfg: dict) -> None:
        wandb.config.update({"env_cfg": env_cfg})
        wandb.config.update({"train_cfg": train_cfg})
        wandb.config.update({"logging_cfg": logging_cfg})

    def _map_path(self, path: str) -> str:
        if path in self.name_map:
            return self.name_map[path]
        else:
            return path

    def add_scalar(
        self,
        tag: str,
        scalar_value: object,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb.log({self._map_path(tag): scalar_value}, step=global_step)

    def finish(self) -> None:
        wandb.finish()

    def log_config(self, env_cfg: dict, train_cfg: dict, logging_cfg: dict) -> None:
        self.store_config(env_cfg, train_cfg, logging_cfg)

    def save(self, path: str) -> None:
        base_path = os.path.dirname(path)
        if os.path.isfile(path):
            wandb.save(path, base_path=base_path)
        elif os.path.isdir(path):
            wandb.save(os.path.join(path, "*"), base_path=base_path)
            wandb.save(os.path.join(path, "**/*"), base_path=base_path)
        else:
            raise ValueError(f"Invalid path: {path}")
