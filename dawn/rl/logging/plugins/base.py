from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


class LoggerPlugin(ABC):
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.enabled = True

    @abstractmethod
    def get_name(self) -> str:
        pass

    def on_register(self, logging_manager: Any) -> None:
        pass

    def initialize_buffers(self, num_envs: int) -> None:
        pass

    def collect_step_metrics(
        self,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict[str, Any],
    ) -> None:
        pass

    def collect_episode_metrics(self, dones: torch.Tensor) -> None:
        pass

    @abstractmethod
    def write_metrics(self, writer: Any, iteration: int, algorithm_metrics: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def format_terminal_output(self, algorithm_metrics: dict[str, Any], pad: int = 35) -> str:
        pass

    def enable(self) -> None:
        self.enabled = True
