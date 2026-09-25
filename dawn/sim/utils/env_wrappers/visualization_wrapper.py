from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from dawn.sim.utils.env_wrappers.vec_env_wrapper import VecEnvWrapper

if TYPE_CHECKING:
    import torch

    from dawn.sim.envs.manager_based_rl_amp_env import ManagerBasedRLAMPEnv


@dataclass
class DepthFigureState:
    figure: Any = None
    axes: list[Any] = field(default_factory=list)
    image_plots: dict[str, Any] = field(default_factory=dict)


class VisualizationWrapper(VecEnvWrapper):
    def __init__(self, env: ManagerBasedRLAMPEnv | VecEnvWrapper):
        super().__init__(env)

        self._color_depth_figure = DepthFigureState()
        self._gray_depth_figure = DepthFigureState()

    @property
    def env_idx(self) -> int:
        return self.env.env_idx

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs_dict, rew, dones, extras = self.env.step(actions)

        if self.unwrapped.cfg.sim.headless:
            return obs_dict, rew, dones, extras

        depth_obs = self._select_depth_observations(obs_dict)
        if not depth_obs:
            return obs_dict, rew, dones, extras

        depth_np = {name: value.cpu().numpy() for name, value in depth_obs.items()}

        self._update_depth_figure(
            self._color_depth_figure,
            depth_np,
            cmap="jet",
        )

        self._update_depth_figure(
            self._gray_depth_figure,
            depth_np,
            cmap="gray",
            title_suffix=" Gray",
        )

        plt.pause(0.001)

        return obs_dict, rew, dones, extras

    def _select_depth_observations(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        depth_obs = {}
        if "depth" in obs_dict:
            depth_obs["Depth"] = obs_dict["depth"][self.env_idx, 0]
        elif "depth_hist" in obs_dict:
            depth_obs["Depth"] = obs_dict["depth_hist"][self.env_idx, -1]

        if "depth_clean" in obs_dict:
            depth_obs["Depth Clean"] = obs_dict["depth_clean"][self.env_idx, 0]

        return depth_obs

    def _update_depth_figure(
        self,
        state: DepthFigureState,
        depth_np: dict[str, Any],
        cmap: str,
        title_suffix: str = "",
    ) -> None:
        if state.figure is None:
            depth_cfg = self.unwrapped.cfg.depth
            plt.ion()
            state.figure, subplot_axes = plt.subplots(1, len(depth_np), figsize=(8 * len(depth_np), 6))
            state.axes = list(subplot_axes) if len(depth_np) > 1 else [subplot_axes]
            for axis, (name, image) in zip(state.axes, depth_np.items(), strict=False):
                image_plot = axis.imshow(
                    image,
                    cmap=cmap,
                    vmin=depth_cfg.near_clip,
                    vmax=depth_cfg.far_clip,
                )
                axis.axis("off")
                state.figure.colorbar(image_plot, ax=axis)
                state.image_plots[name] = image_plot
            state.figure.tight_layout()
        else:
            for name, image in depth_np.items():
                state.image_plots[name].set_data(image)

        for axis, name in zip(state.axes, depth_np, strict=False):
            axis.set_title(f"{name}{title_suffix} (env_idx:{self.env_idx})")
