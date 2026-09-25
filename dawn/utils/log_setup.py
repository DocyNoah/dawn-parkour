from __future__ import annotations

import shlex
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg

    from dawn.config.experiment_cfg import ExperimentCfg
    from dawn.rl.configs.train_cfg import LoggingCfg


def _resolve_log_dir(logging_cfg: LoggingCfg) -> Path:
    log_root_dir = Path("logs") / logging_cfg.experiment_name
    log_root_dir = log_root_dir.resolve()
    print(f"[INFO] Logging experiment in directory: {log_root_dir}")

    log_name = logging_cfg.now_time
    if logging_cfg.run_name:
        log_name = f"{log_name}_{logging_cfg.run_name}"
    return log_root_dir / log_name


def _dump_cmd(file_path: Path, sys_argv: list[str]) -> None:
    try:
        with Path("/proc/self/cmdline").open("rb") as file:
            cmdline_bytes = file.read()
        cmd_args = [arg.decode("utf-8") for arg in cmdline_bytes.split(b"\0") if arg]
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        cmd_args = list(sys_argv)

    cmd = " ".join(shlex.quote(arg) for arg in cmd_args)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(cmd + "\n", encoding="utf-8")


def get_isaaclab_version_path() -> str:
    try:
        import isaaclab.app as isaaclab_app

        isaaclab_app_file = getattr(isaaclab_app, "__file__", None)
        if isaaclab_app_file is None:
            raise RuntimeError("isaaclab.app.__file__ is not available")
        isaaclab_app_path = Path(isaaclab_app_file).resolve()
    except Exception as exc:
        raise RuntimeError(f"Could not find IsaacLab VERSION file: {exc}") from exc

    isaaclab_root_path = isaaclab_app_path.parents[4]
    version_file = isaaclab_root_path / "VERSION"
    if not version_file.is_file():
        raise RuntimeError(f"Could not find IsaacLab VERSION file: VERSION file not found at {version_file}")
    return str(version_file)


def _dump_isaaclab_version(file_path: Path) -> None:
    version_content = Path(get_isaaclab_version_path()).read_text(encoding="utf-8").strip()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(f"IsaacLab VERSION: {version_content}\n", encoding="utf-8")


def _dump_experiment_yaml(file_path: Path, experiment_cfg: ExperimentCfg) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(yaml.safe_dump(asdict(experiment_cfg), sort_keys=False, allow_unicode=True), encoding="utf-8")


def setup_log(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    experiment_cfg: ExperimentCfg,
    sys_argv: list[str],
) -> Path:
    from isaaclab.utils.io import dump_yaml

    logging_cfg = experiment_cfg.logging
    log_dir = _resolve_log_dir(logging_cfg)
    params_dir = log_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    _dump_experiment_yaml(params_dir / "experiment.yaml", experiment_cfg)

    dump_yaml(str(params_dir / "env.yaml"), env_cfg)

    _dump_cmd(log_dir / "cmd.txt", sys_argv)

    _dump_isaaclab_version(log_dir / "isaaclab_version.txt")
    return log_dir
