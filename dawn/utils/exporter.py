import copy
import os

import torch
import yaml
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from torch import nn

from dawn.rl.networks.base import BaseNetwork


def export_policy_as_jit(
    actor: nn.Module,
    path: str,
    filename: str = "policy.pt",
    device: str = "cpu",
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | None = None,
) -> None:
    policy_exporter = _TorchPolicyExporter(actor, device)
    policy_exporter.export(path, filename)

    if env_cfg is not None:
        ranges = env_cfg.commands.base_velocity.ranges
        yaml_path = os.path.join(path, filename.replace(".pt", ".yaml"))
        os.makedirs(path, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(
                {
                    "velocity_commands": {
                        "x": {"min": float(ranges.lin_vel_x[0]), "max": float(ranges.lin_vel_x[1])},
                        "y": {"min": float(ranges.lin_vel_y[0]), "max": float(ranges.lin_vel_y[1])},
                        "z": {"min": float(ranges.ang_vel_z[0]), "max": float(ranges.ang_vel_z[1])},
                    }
                },
                f,
                default_flow_style=False,
                sort_keys=False,
            )


class _TorchPolicyExporter(torch.nn.Module):
    def __init__(
        self,
        actor: BaseNetwork,
        device: str = "cpu",
    ):
        super().__init__()
        self.actor = copy.deepcopy(actor)
        self.device = device

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.actor(x)

    @torch.jit.export
    def reset(self) -> None:
        pass

    def export(self, path: str, filename: str) -> None:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to(self.device)
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
