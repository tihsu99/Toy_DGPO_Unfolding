from __future__ import annotations

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
    required = {
        "physics", "detector", "data", "flow", "training", "fisher_validation",
        "dgpo", "refresh", "inference", "diagnosis", "closure", "policies", "plots",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    if int(config["dgpo"]["group_size"]) < 2:
        raise ValueError("dgpo.group_size must be at least two")
    if int(config["refresh"]["rounds"]) < 1:
        raise ValueError("refresh.rounds must be positive")
    if int(config["refresh"]["dgpo_epochs_per_round"]) < 1:
        raise ValueError("refresh.dgpo_epochs_per_round must be positive")
    if int(config["refresh"]["direct_fisher_bins"]) < 2:
        raise ValueError("refresh.direct_fisher_bins must be at least two")
    if not config["policies"].get("baseline", False):
        raise ValueError("policies.baseline must be enabled because it defines the frozen reference")
    if not 0.0 < float(config["physics"]["nominal_C"]) < 1.0:
        raise ValueError("physics.nominal_C must lie in (0, 1)")
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


def truth_score(x: torch.Tensor, nominal_C: float) -> torch.Tensor:
    return x / (1.0 + nominal_C * x)


class ScoreModel(nn.Module):
    def __init__(self, width: int, layers: int):
        super().__init__()
        modules: list[nn.Module] = []
        current = 1
        for _ in range(layers):
            modules.extend((nn.Linear(current, width), nn.SiLU()))
            current = width
        modules.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*modules)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.network(y.reshape(-1, 1)).reshape(y.shape)
