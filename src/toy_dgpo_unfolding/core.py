from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    required = {"physics", "data", "model", "training", "unfolding", "policies", "plots"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    return config


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def sample_truth(count: int, C: float, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    if abs(C) >= 1.0:
        raise ValueError("The linear truth density requires |C| < 1")
    u = torch.rand(count, device=device, generator=generator)
    if abs(C) < 1.0e-10:
        return 2.0 * u - 1.0
    return (-1.0 + torch.sqrt((1.0 - C) ** 2 + 4.0 * C * u)) / C


def detector_features(
    truth: torch.Tensor,
    physics: dict[str, Any],
    generator: torch.Generator,
) -> torch.Tensor:
    sigma = float(physics["detector_resolution"])
    aux_sigma = float(physics["auxiliary_resolution"])
    bias = float(physics["detector_bias"])
    noise = torch.randn((truth.numel(), 3), device=truth.device, generator=generator)
    return torch.stack(
        (
            truth + bias + sigma * noise[:, 0],
            0.5 * truth.square() + aux_sigma * noise[:, 1],
            torch.sin(torch.pi * truth) + aux_sigma * noise[:, 2],
        ),
        dim=1,
    )


def nominal_score(truth: torch.Tensor, nominal_C: float) -> torch.Tensor:
    return truth / (1.0 + nominal_C * truth)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int, layers: int, bounded: bool = False):
        super().__init__()
        modules: list[nn.Module] = []
        current = input_dim
        for _ in range(layers):
            modules.extend((nn.Linear(current, width), nn.SiLU()))
            current = width
        modules.append(nn.Linear(current, output_dim))
        if bounded:
            modules.append(nn.Tanh())
        self.network = nn.Sequential(*modules)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


def make_policy(config: dict[str, Any]) -> MLP:
    model = config["model"]
    return MLP(3, 1, int(model["hidden_width"]), int(model["hidden_layers"]), bounded=True)


def make_score_model(config: dict[str, Any]) -> MLP:
    model = config["model"]
    return MLP(1, 1, int(model["hidden_width"]), int(model["hidden_layers"]), bounded=False)


def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


@torch.no_grad()
def reconstruct(
    policy: nn.Module,
    features: torch.Tensor,
    sigma: float,
    generator: torch.Generator,
) -> torch.Tensor:
    mean = policy(features)
    if sigma > 0.0:
        mean = mean + sigma * torch.randn(mean.shape, device=mean.device, generator=generator)
    return mean.clamp(-1.0, 1.0)
