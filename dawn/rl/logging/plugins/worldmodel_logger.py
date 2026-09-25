from __future__ import annotations

from typing import Any

from dawn.rl.logging.plugins.base import LoggerPlugin


class WorldModelLogger(LoggerPlugin):
    def get_name(self) -> str:
        return "worldmodel"

    def write_metrics(self, writer: Any, iteration: int, algorithm_metrics: dict[str, Any]) -> None:
        if "rep_scale" in algorithm_metrics and algorithm_metrics["rep_scale"] is not None:
            writer.add_scalar("wm/rep_scale", algorithm_metrics["rep_scale"], iteration)

        if "rep_loss" in algorithm_metrics and algorithm_metrics["rep_loss"] is not None:
            writer.add_scalar("wm/rep_loss", algorithm_metrics["rep_loss"], iteration)

        if "prop_loss" in algorithm_metrics and algorithm_metrics["prop_loss"] is not None:
            writer.add_scalar("wm/prop_loss", algorithm_metrics["prop_loss"], iteration)

        if "prior_ent" in algorithm_metrics and algorithm_metrics["prior_ent"] is not None:
            writer.add_scalar("wm/prior_ent", algorithm_metrics["prior_ent"], iteration)

        if "post_ent" in algorithm_metrics and algorithm_metrics["post_ent"] is not None:
            writer.add_scalar("wm/post_ent", algorithm_metrics["post_ent"], iteration)

        if "model_loss" in algorithm_metrics and algorithm_metrics["model_loss"] is not None:
            writer.add_scalar("wm/model_loss", algorithm_metrics["model_loss"], iteration)

        if "model_grad_norm" in algorithm_metrics and algorithm_metrics["model_grad_norm"] is not None:
            writer.add_scalar("wm/model_grad_norm", algorithm_metrics["model_grad_norm"], iteration)

        if "kl_free" in algorithm_metrics and algorithm_metrics["kl_free"] is not None:
            writer.add_scalar("wm/kl_free", algorithm_metrics["kl_free"], iteration)

        if "kl" in algorithm_metrics and algorithm_metrics["kl"] is not None:
            writer.add_scalar("wm/kl", algorithm_metrics["kl"], iteration)

        if "image_loss" in algorithm_metrics and algorithm_metrics["image_loss"] is not None:
            writer.add_scalar("wm/image_loss", algorithm_metrics["image_loss"], iteration)

        if "dyn_scale" in algorithm_metrics and algorithm_metrics["dyn_scale"] is not None:
            writer.add_scalar("wm/dyn_scale", algorithm_metrics["dyn_scale"], iteration)

        if "dyn_loss" in algorithm_metrics and algorithm_metrics["dyn_loss"] is not None:
            writer.add_scalar("wm/dyn_loss", algorithm_metrics["dyn_loss"], iteration)

        if "contrastive_loss" in algorithm_metrics and algorithm_metrics["contrastive_loss"] is not None:
            writer.add_scalar("wm/contrastive_loss", algorithm_metrics["contrastive_loss"], iteration)

    def format_terminal_output(self, algorithm_metrics: dict[str, Any], pad: int = 35) -> str:
        return ""
