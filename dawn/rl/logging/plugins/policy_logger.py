from __future__ import annotations

from typing import Any

from dawn.rl.logging.plugins.base import LoggerPlugin


class PolicyLogger(LoggerPlugin):
    def get_name(self) -> str:
        return "policy"

    def write_metrics(self, writer: Any, iteration: int, algorithm_metrics: dict[str, Any]) -> None:
        if "mean_value_loss" in algorithm_metrics and algorithm_metrics["mean_value_loss"] is not None:
            writer.add_scalar("Loss/value_function", algorithm_metrics["mean_value_loss"], iteration)

        if "mean_surrogate_loss" in algorithm_metrics and algorithm_metrics["mean_surrogate_loss"] is not None:
            writer.add_scalar("Loss/surrogate", algorithm_metrics["mean_surrogate_loss"], iteration)

        if "mean_entropy" in algorithm_metrics and algorithm_metrics["mean_entropy"] is not None:
            writer.add_scalar("Loss/entropy", algorithm_metrics["mean_entropy"], iteration)

        if "learning_rate" in algorithm_metrics and algorithm_metrics["learning_rate"] is not None:
            writer.add_scalar("Loss/learning_rate", algorithm_metrics["learning_rate"], iteration)

        if "mean_vel_predict_loss" in algorithm_metrics and algorithm_metrics["mean_vel_predict_loss"] is not None:
            writer.add_scalar("Loss/vel_predict", algorithm_metrics["mean_vel_predict_loss"], iteration)

        if "mean_amp_loss" in algorithm_metrics and algorithm_metrics["mean_amp_loss"] is not None:
            writer.add_scalar("Loss/AMP", algorithm_metrics["mean_amp_loss"], iteration)

        if "mean_grad_pen_loss" in algorithm_metrics and algorithm_metrics["mean_grad_pen_loss"] is not None:
            writer.add_scalar("Loss/AMP_grad", algorithm_metrics["mean_grad_pen_loss"], iteration)

        if "mean_policy_pred" in algorithm_metrics and algorithm_metrics["mean_policy_pred"] is not None:
            writer.add_scalar("Loss/AMP_mean_policy_pred", algorithm_metrics["mean_policy_pred"], iteration)

        if "mean_expert_pred" in algorithm_metrics and algorithm_metrics["mean_expert_pred"] is not None:
            writer.add_scalar("Loss/AMP_mean_expert_pred", algorithm_metrics["mean_expert_pred"], iteration)

        if "mean_noise_std" in algorithm_metrics and algorithm_metrics["mean_noise_std"] is not None:
            writer.add_scalar("Policy/mean_noise_std", algorithm_metrics["mean_noise_std"], iteration)

    def format_terminal_output(self, algorithm_metrics: dict[str, Any], pad: int = 35) -> str:
        output = ""

        if "mean_value_loss" in algorithm_metrics and algorithm_metrics["mean_value_loss"] is not None:
            output += f"""{"Value function loss:":>{pad}} {algorithm_metrics["mean_value_loss"]:.4f}\n"""

        if "mean_surrogate_loss" in algorithm_metrics and algorithm_metrics["mean_surrogate_loss"] is not None:
            output += f"""{"Surrogate loss:":>{pad}} {algorithm_metrics["mean_surrogate_loss"]:.4f}\n"""

        if "mean_noise_std" in algorithm_metrics and algorithm_metrics["mean_noise_std"] is not None:
            output += f"""{"Mean action noise std:":>{pad}} {algorithm_metrics["mean_noise_std"]:.2f}\n"""

        if "mean_vel_predict_loss" in algorithm_metrics and algorithm_metrics["mean_vel_predict_loss"] is not None:
            output += f"""{"Vel predict loss:":>{pad}} {algorithm_metrics["mean_vel_predict_loss"]:.4f}\n"""

        if "mean_amp_loss" in algorithm_metrics and algorithm_metrics["mean_amp_loss"] is not None:
            output += f"""{"AMP loss:":>{pad}} {algorithm_metrics["mean_amp_loss"]:.4f}\n"""

        if "mean_grad_pen_loss" in algorithm_metrics and algorithm_metrics["mean_grad_pen_loss"] is not None:
            output += f"""{"AMP grad pen loss:":>{pad}} {algorithm_metrics["mean_grad_pen_loss"]:.4f}\n"""

        if "mean_policy_pred" in algorithm_metrics and algorithm_metrics["mean_policy_pred"] is not None:
            output += f"""{"AMP mean policy pred:":>{pad}} {algorithm_metrics["mean_policy_pred"]:.4f}\n"""

        if "mean_expert_pred" in algorithm_metrics and algorithm_metrics["mean_expert_pred"] is not None:
            output += f"""{"AMP mean expert pred:":>{pad}} {algorithm_metrics["mean_expert_pred"]:.4f}\n"""

        return output
