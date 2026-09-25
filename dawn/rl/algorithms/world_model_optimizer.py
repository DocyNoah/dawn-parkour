from typing import Any

import torch
from torch import nn


class RequiresGrad:
    def __init__(self, model: nn.Module) -> None:
        self._model = model

    def __enter__(self) -> None:
        self._model.requires_grad_(requires_grad=True)

    def __exit__(self, *args: Any) -> None:
        self._model.requires_grad_(requires_grad=False)


class Optimizer:
    def __init__(
        self,
        name: str,
        parameters: Any,
        lr: float,
        eps: float = 1e-4,
        clip: float | None = None,
        wd: float | None = None,
        use_amp: bool = False,
    ) -> None:
        assert 0 <= wd < 1
        assert not clip or clip >= 1
        self._name = name
        self._parameters = parameters
        self._clip = clip
        self._wd = wd
        self._opt = torch.optim.Adam(parameters, lr=lr, eps=eps)
        self._scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def zero_grad(self) -> None:
        self._opt.zero_grad()

    def add_parameters(self, parameters: Any) -> None:
        self._opt.add_param_group({"params": parameters})

    def backward(self, loss: torch.Tensor, retain_graph: bool = True) -> None:
        self._scaler.scale(loss).backward(retain_graph=retain_graph)

    def step(self, loss_value: float, params: Any) -> dict[str, float]:
        metrics = {}
        metrics[f"{self._name}_loss"] = loss_value
        self._scaler.unscale_(self._opt)
        norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
        if self._wd:
            self._apply_weight_decay(params)
        self._scaler.step(self._opt)
        self._scaler.update()
        self._opt.zero_grad()
        metrics[f"{self._name}_grad_norm"] = norm.item()
        return metrics

    def _apply_weight_decay(self, varibs: Any) -> None:
        for var in varibs:
            var.data = (1 - self._wd) * var.data
