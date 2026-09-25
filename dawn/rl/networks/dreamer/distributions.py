from typing import Any

import torch
import torch.nn.functional as F
from torch import distributions as torchd


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(
        self,
        logits: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        unimix_ratio: float = 0.0,
    ) -> None:
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self) -> torch.Tensor:
        mode_ = F.one_hot(torch.argmax(super().logits, axis=-1), super().logits.shape[-1])
        return mode_.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape: tuple = (), seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape)

        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample


class DiscDist:
    def __init__(
        self,
        logits: torch.Tensor,
        low: float = -20.0,
        high: float = 20.0,
        transfwd: Any = symlog,
        transbwd: Any = symexp,
        device: str | torch.device = "cuda",
    ) -> None:
        self.logits = logits
        self.probs = torch.softmax(logits, -1)

        self.buckets = torch.linspace(low, high, steps=255).to(device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255
        self.transfwd = transfwd
        self.transbwd = transbwd

    def mean(self) -> torch.Tensor:
        mean_ = self.probs * self.buckets
        return self.transbwd(torch.sum(mean_, dim=-1, keepdim=True))

    def mode(self) -> torch.Tensor:
        mode_ = self.probs * self.buckets
        return self.transbwd(torch.sum(mode_, dim=-1, keepdim=True))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transfwd(x)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum((self.buckets > x[..., None]).to(torch.int32), dim=-1)
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )

        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)


class MSEDist:
    def __init__(self, mode: torch.Tensor) -> None:
        self._mode = mode

    def mode(self) -> torch.Tensor:
        return self._mode

    def mean(self) -> torch.Tensor:
        return self._mode

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        loss = distance.sum(list(range(len(distance.shape)))[2:])
        return -loss


class SymlogDist:
    def __init__(self, mode: torch.Tensor, tol: float = 1e-8) -> None:
        self._mode = mode
        self._tol = tol

    def mode(self) -> torch.Tensor:
        return symexp(self._mode)

    def mean(self) -> torch.Tensor:
        return symexp(self._mode)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        assert self._mode.shape == value.shape
        distance = (self._mode - symlog(value)) ** 2.0
        distance = torch.where(distance < self._tol, 0, distance)
        loss = distance.sum(list(range(len(distance.shape)))[2:])
        return -loss


class ContDist:
    def __init__(self, dist: Any = None) -> None:
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dist, name)

    def entropy(self) -> torch.Tensor:
        return self._dist.entropy()

    def mode(self) -> torch.Tensor:
        return self._dist.mean

    def sample(self, sample_shape: tuple = ()) -> torch.Tensor:
        return self._dist.rsample(sample_shape)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self._dist.log_prob(x)


class Bernoulli:
    def __init__(self, dist: Any = None) -> None:
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dist, name)

    def entropy(self) -> torch.Tensor:
        return self._dist.entropy()

    def mode(self) -> torch.Tensor:
        mode_ = torch.round(self._dist.mean)
        return mode_.detach() + self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape: tuple = ()) -> torch.Tensor:
        return self._dist.rsample(sample_shape)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        logits_ = self._dist.base_dist.logits
        log_probs0 = -F.softplus(logits_)
        log_probs1 = -F.softplus(-logits_)

        return torch.sum(log_probs0 * (1 - x) + log_probs1 * x, -1)
