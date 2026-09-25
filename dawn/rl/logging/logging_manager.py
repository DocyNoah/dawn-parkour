from __future__ import annotations

import statistics
from collections import deque
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from dawn.rl.configs.train_cfg import LoggingCfg
    from dawn.rl.logging.plugins.base import LoggerPlugin


class LoggingManager:
    def __init__(
        self,
        log_dir: str,
        device: str,
        buffer_size: int = 100,
    ):
        self.log_dir = log_dir
        self.device = device
        self.buffer_size = buffer_size

        self.writer = None
        self.logger_type = None

        self.plugins: dict[str, LoggerPlugin] = {}

        self.rewbuffer: deque = deque(maxlen=buffer_size)
        self.lenbuffer: deque = deque(maxlen=buffer_size)
        self.terrain_rewbuffer: dict[str, deque] = {}
        self.terrain_lenbuffer: dict[str, deque] = {}
        self.terrain_successbuffer: dict[str, deque] = {}
        self._last_terrain_type_names: list[str] | None = None
        self._last_success_flags: torch.Tensor | None = None

        self.action_min_buffer: deque = deque(maxlen=buffer_size)
        self.action_max_buffer: deque = deque(maxlen=buffer_size)
        self.action_mean_buffer: deque = deque(maxlen=buffer_size)
        self.action_std_buffer: deque = deque(maxlen=buffer_size)

        self.ep_infos: list = []

        self.cur_reward_sum: torch.Tensor | None = None
        self.cur_episode_length: torch.Tensor | None = None

        self.tot_timesteps = 0
        self.tot_time = 0.0

    def register_plugin(self, plugin: LoggerPlugin) -> None:
        name = plugin.get_name()
        if name in self.plugins:
            raise ValueError(f"Plugin '{name}' already registered")
        self.plugins[name] = plugin
        plugin.on_register(self)

    def initialize_env_tracking(self, num_envs: int) -> None:
        self.cur_reward_sum = torch.zeros(num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(num_envs, dtype=torch.float, device=self.device)

        for plugin in self.plugins.values():
            if plugin.enabled:
                plugin.initialize_buffers(num_envs)

    def initialize_writer(
        self,
        logging_cfg: LoggingCfg,
        env_cfg_dict: dict,
        train_cfg_dict: dict,
    ) -> None:
        if self.writer is not None:
            return

        self.logger_type = logging_cfg.logger.lower()
        logging_cfg_dict = logging_cfg.to_dict()

        if self.logger_type == "wandb":
            from dawn.rl.utils.wandb_utils import WandbSummaryWriter

            self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=logging_cfg_dict)
            self.writer.log_config(
                env_cfg_dict,
                train_cfg_dict,
                logging_cfg_dict,
            )
        elif self.logger_type == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        else:
            raise ValueError("Logger type not found. Please choose 'wandb' or 'tensorboard'.")

    def collect_step_metrics(
        self,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict[str, Any],
    ) -> None:
        if "episode" in infos:
            self.ep_infos.append(infos["episode"])
        elif "log" in infos:
            self.ep_infos.append(infos["log"])

        self.action_min_buffer.append(actions.min().item())
        self.action_max_buffer.append(actions.max().item())
        self.action_mean_buffer.append(actions.mean().item())
        self.action_std_buffer.append(actions.std().item())

        cur_reward_sum, cur_episode_length = self._require_episode_tracking()
        cur_reward_sum += rewards
        cur_episode_length += 1
        terrain_type_names = infos.get("terrain_type_names")
        if terrain_type_names is not None:
            self._last_terrain_type_names = list(terrain_type_names)
        success_flags = infos.get("time_outs")
        if success_flags is not None:
            self._last_success_flags = success_flags.detach().to(self.device).bool()

        for plugin in self.plugins.values():
            if plugin.enabled:
                plugin.collect_step_metrics(actions, rewards, dones, infos)

    def collect_episode_metrics(self, dones: torch.Tensor) -> None:
        new_ids = (dones > 0).nonzero(as_tuple=False)
        cur_reward_sum, cur_episode_length = self._require_episode_tracking()

        if len(new_ids) > 0:
            self.rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            self.lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())

            self._collect_terrain_episode_metrics(new_ids, cur_reward_sum, cur_episode_length)

            cur_reward_sum[new_ids] = 0
            cur_episode_length[new_ids] = 0

            for plugin in self.plugins.values():
                if plugin.enabled:
                    plugin.collect_episode_metrics(dones)

    def _collect_terrain_episode_metrics(
        self,
        done_ids: torch.Tensor,
        cur_reward_sum: torch.Tensor,
        cur_episode_length: torch.Tensor,
    ) -> None:
        if self._last_terrain_type_names is None:
            return

        for done_id in done_ids.flatten().tolist():
            if done_id >= len(self._last_terrain_type_names):
                continue
            terrain_name = self._last_terrain_type_names[done_id]
            self.terrain_rewbuffer.setdefault(terrain_name, deque(maxlen=self.buffer_size)).append(
                cur_reward_sum[done_id].item()
            )
            self.terrain_lenbuffer.setdefault(terrain_name, deque(maxlen=self.buffer_size)).append(
                cur_episode_length[done_id].item()
            )
            if self._last_success_flags is not None and done_id < self._last_success_flags.shape[0]:
                self.terrain_successbuffer.setdefault(terrain_name, deque(maxlen=self.buffer_size)).append(
                    float(self._last_success_flags[done_id].item())
                )

    def log(
        self,
        iteration: int,
        total_iterations: int,
        collection_time: float,
        learn_time: float,
        num_steps_per_env: int,
        num_envs: int,
        algorithm_metrics: dict[str, Any],
    ) -> None:
        extra_time = self._get_extra_time(algorithm_metrics)
        extra_time_name = self._get_extra_time_name(algorithm_metrics)

        self.tot_timesteps += num_steps_per_env * num_envs
        self.tot_time += collection_time + learn_time + extra_time
        iteration_time = collection_time + learn_time + extra_time

        fps = int(num_steps_per_env * num_envs / iteration_time)

        self._write_episode_metrics(iteration)
        self._write_terrain_episode_metrics(iteration)
        self._write_action_metrics(iteration)
        self._write_performance_metrics(
            iteration,
            fps,
            collection_time,
            learn_time,
            extra_time,
            iteration_time,
            extra_time_name,
        )
        self._write_training_metrics(iteration)
        writer = self._require_writer()

        for plugin in self.plugins.values():
            if plugin.enabled:
                plugin.write_metrics(writer, iteration, algorithm_metrics)

        terminal_output = self._format_terminal_output(
            iteration=iteration,
            total_iterations=total_iterations,
            fps=fps,
            collection_time=collection_time,
            learn_time=learn_time,
            iteration_time=iteration_time,
            extra_time=extra_time,
            extra_time_name=extra_time_name,
            algorithm_metrics=algorithm_metrics,
        )
        print(terminal_output)

        self.ep_infos.clear()

    def _write_episode_metrics(self, iteration: int) -> None:
        if not self.ep_infos:
            return

        writer = self._require_writer()

        for key in self.ep_infos[0]:
            infotensor = torch.tensor([], device=self.device)
            for ep_info in self.ep_infos:
                if key not in ep_info:
                    continue
                if not isinstance(ep_info[key], torch.Tensor):
                    ep_info[key] = torch.Tensor([ep_info[key]])
                if len(ep_info[key].shape) == 0:
                    ep_info[key] = ep_info[key].unsqueeze(0)
                infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))

            value = torch.mean(infotensor)

            if "/" in key:
                writer.add_scalar(key, value, iteration)
            else:
                writer.add_scalar("Episode/" + key, value, iteration)

    def _write_terrain_episode_metrics(self, iteration: int) -> None:
        writer = self._require_writer()
        for terrain_name, reward_buffer in self.terrain_rewbuffer.items():
            if reward_buffer:
                writer.add_scalar(
                    f"TerrainEpisode/{terrain_name}/mean_reward",
                    statistics.mean(reward_buffer),
                    iteration,
                )
        for terrain_name, length_buffer in self.terrain_lenbuffer.items():
            if length_buffer:
                writer.add_scalar(
                    f"TerrainEpisode/{terrain_name}/mean_episode_length",
                    statistics.mean(length_buffer),
                    iteration,
                )
        for terrain_name, success_buffer in self.terrain_successbuffer.items():
            if success_buffer:
                writer.add_scalar(
                    f"TerrainEpisode/{terrain_name}/success_rate",
                    statistics.mean(success_buffer),
                    iteration,
                )

    def _write_action_metrics(self, iteration: int) -> None:
        if len(self.action_min_buffer) == 0:
            return

        writer = self._require_writer()

        writer.add_scalar("Actions/min", statistics.mean(self.action_min_buffer), iteration)
        writer.add_scalar("Actions/max", statistics.mean(self.action_max_buffer), iteration)
        writer.add_scalar("Actions/mean", statistics.mean(self.action_mean_buffer), iteration)
        writer.add_scalar("Actions/std", statistics.mean(self.action_std_buffer), iteration)

    def _write_performance_metrics(
        self,
        iteration: int,
        fps: int,
        collection_time: float,
        learn_time: float,
        extra_time: float,
        iteration_time: float,
        extra_time_name: str,
    ) -> None:
        writer = self._require_writer()
        writer.add_scalar("Perf/total_fps", fps, iteration)
        writer.add_scalar("Perf/collection time", collection_time, iteration)
        writer.add_scalar("Perf/learning_time", learn_time, iteration)
        writer.add_scalar("Perf/extra_time", extra_time, iteration)
        writer.add_scalar(f"Perf/{extra_time_name.replace(' ', '_')}_time", extra_time, iteration)
        writer.add_scalar("Perf/iteration_time", iteration_time, iteration)
        writer.add_scalar("Perf/total_timesteps", self.tot_timesteps, iteration)

    def _write_training_metrics(self, iteration: int) -> None:
        if len(self.rewbuffer) == 0:
            return

        writer = self._require_writer()

        writer.add_scalar("Train/mean_reward", statistics.mean(self.rewbuffer), iteration)
        writer.add_scalar("Train/mean_episode_length", statistics.mean(self.lenbuffer), iteration)

        if self.logger_type != "wandb":
            writer.add_scalar("Train/mean_reward/time", statistics.mean(self.rewbuffer), self.tot_time)
            writer.add_scalar("Train/mean_episode_length/time", statistics.mean(self.lenbuffer), self.tot_time)

    def _format_terminal_output(
        self,
        iteration: int,
        total_iterations: int,
        fps: int,
        collection_time: float,
        learn_time: float,
        iteration_time: float,
        extra_time: float,
        extra_time_name: str,
        algorithm_metrics: dict[str, Any],
        width: int = 120,
        pad: int = 70,
    ) -> str:
        ep_string = ""
        if self.ep_infos:
            for key in self.ep_infos[0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in self.ep_infos:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)

                if "/" in key:
                    ep_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
                else:
                    ep_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""

        extra_time_string = ""
        if extra_time > 0.0:
            extra_time_string = f", {extra_time_name}: {extra_time:.3f}s"

        log_string = (
            f"""{"#" * width}\n"""
            f"""\033[1m{"Learning iteration":>{pad}} {iteration}/{total_iterations}\033[0m\n\n"""
            f"""{"Computation:":>{pad}} {fps:.0f} """
            f"""steps/s (collection: {collection_time:.3f}s, """
            f"""learning {learn_time:.3f}s{extra_time_string})\n"""
        )

        for plugin in self.plugins.values():
            if plugin.enabled:
                plugin_output = plugin.format_terminal_output(algorithm_metrics, pad)
                if plugin_output:
                    log_string += plugin_output

        if len(self.rewbuffer) > 0:
            log_string += f"""{"Mean total reward:":>{pad}} {statistics.mean(self.rewbuffer):.2f}\n"""
            log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(self.lenbuffer):.2f}\n"""

            if len(self.action_min_buffer) > 0:
                log_string += f"""{"Mean action min:":>{pad}} {statistics.mean(self.action_min_buffer):.4f}\n"""
                log_string += f"""{"Mean action max:":>{pad}} {statistics.mean(self.action_max_buffer):.4f}\n"""
                log_string += f"""{"Mean action mean:":>{pad}} {statistics.mean(self.action_mean_buffer):.4f}\n"""
                log_string += f"""{"Mean action std:":>{pad}} {statistics.mean(self.action_std_buffer):.4f}\n"""

        log_string += ep_string

        start_iter = iteration - (iteration % 1)
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Total time:":>{pad}} {self.tot_time:.2f}s\n"""
            f"""{"ETA:":>{pad}} """
            f"""{self.tot_time / max(1, iteration - start_iter + 1) * (total_iterations - iteration):.1f}s\n"""
        )

        return log_string

    def _get_extra_time(self, algorithm_metrics: dict[str, Any]) -> float:
        extra_time = algorithm_metrics.get("extra_time", 0.0)
        if isinstance(extra_time, torch.Tensor):
            return float(extra_time.item())
        if isinstance(extra_time, int | float):
            return float(extra_time)
        return 0.0

    def _get_extra_time_name(self, algorithm_metrics: dict[str, Any]) -> str:
        extra_time_name = algorithm_metrics.get("extra_time_name", "extra")
        if isinstance(extra_time_name, str):
            return extra_time_name
        return "extra"

    def finish(self) -> None:
        if self.logger_type == "wandb":
            self._require_writer().finish()

    def save(self, path: str) -> None:
        if self.logger_type == "wandb":
            self._require_writer().save(path)

    def _require_episode_tracking(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cur_reward_sum is None or self.cur_episode_length is None:
            raise RuntimeError("Environment tracking must be initialized before collecting episode metrics.")
        return self.cur_reward_sum, self.cur_episode_length

    def _require_writer(self) -> Any:
        if self.writer is None:
            raise RuntimeError("Writer must be initialized before logging metrics.")
        return self.writer
