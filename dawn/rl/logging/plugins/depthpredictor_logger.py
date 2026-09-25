from __future__ import annotations

from typing import Any

from dawn.rl.logging.plugins.base import LoggerPlugin


class DepthPredictorLogger(LoggerPlugin):
    def get_name(self) -> str:
        return "depthpredictor"

    def write_metrics(self, writer: Any, iteration: int, algorithm_metrics: dict[str, Any]) -> None:
        if "depth_predictor_loss" in algorithm_metrics and algorithm_metrics["depth_predictor_loss"] is not None:
            writer.add_scalar("DepthPredictor/loss", algorithm_metrics["depth_predictor_loss"], iteration)

    def format_terminal_output(self, algorithm_metrics: dict[str, Any], pad: int = 35) -> str:
        return ""
