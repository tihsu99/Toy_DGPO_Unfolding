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
        "dgpo", "refresh", "ablation", "inference", "diagnosis", "closure", "policies", "plots",
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
    if config["ablation"].get("enabled", False):
        required_policies = set(config["ablation"]["policy_order"])
        disabled = sorted(name for name in required_policies if not config["policies"].get(name, False))
        if disabled:
            raise ValueError(f"The enabled ablation requires these policies: {disabled}")
        frozen_epochs = int(config["dgpo"]["epochs"])
        refresh_epochs = int(config["refresh"]["rounds"]) * int(config["refresh"]["dgpo_epochs_per_round"])
        if frozen_epochs != refresh_epochs:
            raise ValueError("The ablation requires equal frozen and iterative DGPO epoch budgets")
        if float(config["refresh"]["beta_global"]) != 0.0:
            raise ValueError("The clean ablation requires refresh.beta_global = 0")
        bin_counts = [int(value) for value in config["ablation"]["fisher_bin_counts"]]
        if not bin_counts or max(bin_counts) < 80:
            raise ValueError("The ablation requires a direct Fisher result with at least 80 bins")
        if int(config["ablation"]["selection_bins"]) not in bin_counts:
            raise ValueError("ablation.selection_bins must be present in ablation.fisher_bin_counts")
        if int(config["ablation"]["validation_events"]) < 1:
            raise ValueError("ablation.validation_events must be positive")
        if int(config["ablation"]["frozen_validation_interval_epochs"]) < 1:
            raise ValueError("ablation.frozen_validation_interval_epochs must be positive")
        if int(config["ablation"]["early_stop_patience_rounds"]) < 1:
            raise ValueError("ablation.early_stop_patience_rounds must be positive")
        if float(config["ablation"]["min_relative_fisher_improvement"]) < 0.0:
            raise ValueError("ablation.min_relative_fisher_improvement must be non-negative")
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


def comparison_policy_names(config: dict[str, Any], available: Any) -> list[str]:
    """Return enabled comparison policies in configured order and reject partial plots."""
    available_names = set(available)
    names = [
        str(name) for name in config["ablation"]["policy_order"]
        if config["policies"].get(name, False)
    ]
    missing = [name for name in names if name not in available_names]
    if missing:
        raise RuntimeError(f"Cannot make a complete policy comparison; missing: {missing}")
    return names


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
