from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .core import clone_model, make_policy, make_score_model, nominal_score


def _loader(*tensors: torch.Tensor, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(TensorDataset(*[tensor.cpu() for tensor in tensors]), batch_size=batch_size, shuffle=shuffle)


def train_baseline(
    features: torch.Tensor,
    truth: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    settings = config["training"]
    model = make_policy(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"])
    )
    loader = _loader(features, truth, batch_size=int(settings["batch_size"]))
    progress = tqdm(range(int(settings["baseline_epochs"])), desc="baseline", unit="epoch")
    for _ in progress:
        total = 0.0
        seen = 0
        for batch_features, batch_truth in loader:
            batch_features, batch_truth = batch_features.to(device), batch_truth.to(device)
            loss = torch.mean((model(batch_features) - batch_truth).square())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["grad_clip_norm"]))
            optimizer.step()
            total += loss.item() * batch_truth.numel()
            seen += batch_truth.numel()
        progress.set_postfix(mse=f"{total / seen:.5f}")
    return model.eval()


def train_score_model(
    baseline: nn.Module,
    features: torch.Tensor,
    truth: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    settings = config["training"]
    sigma = float(config["model"]["policy_sigma"])
    with torch.no_grad():
        reco = (baseline(features.to(device)) + sigma * torch.randn(truth.numel(), device=device)).clamp(-1.0, 1.0)
    targets = nominal_score(truth.to(device), float(config["physics"]["nominal_C"]))
    model = make_score_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"])
    )
    loader = _loader(reco[:, None], targets, batch_size=int(settings["batch_size"]))
    progress = tqdm(range(int(settings["score_epochs"])), desc="score model", unit="epoch")
    for _ in progress:
        total = 0.0
        seen = 0
        for batch_reco, batch_target in loader:
            batch_reco, batch_target = batch_reco.to(device), batch_target.to(device)
            loss = torch.mean((model(batch_reco) - batch_target).square())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * batch_target.numel()
            seen += batch_target.numel()
        progress.set_postfix(mse=f"{total / seen:.5f}")
    return model.eval()


def train_dgpo(
    baseline: nn.Module,
    score_model: nn.Module,
    features: torch.Tensor,
    truth: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    bias_controlled: bool,
) -> nn.Module:
    settings = config["training"]
    policy = clone_model(baseline).to(device).train()
    frozen_baseline = clone_model(baseline).to(device).eval()
    score_model.eval()
    for parameter in frozen_baseline.parameters():
        parameter.requires_grad_(False)
    for parameter in score_model.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"])
    )
    loader = _loader(features, truth, batch_size=int(settings["batch_size"]))
    group_size = int(settings["group_size"])
    sigma = float(config["model"]["policy_sigma"])
    clip = float(settings["clip_epsilon"])
    label = "bias-controlled DGPO" if bias_controlled else "Fisher DGPO"
    progress = tqdm(range(int(settings["dgpo_epochs"])), desc=label, unit="epoch")
    for _ in progress:
        epoch_reward = 0.0
        batches = 0
        for batch_features, batch_truth in loader:
            batch_features, batch_truth = batch_features.to(device), batch_truth.to(device)
            with torch.no_grad():
                old_mean = policy(batch_features)
                baseline_mean = frozen_baseline(batch_features)
                actions = (old_mean[:, None] + sigma * torch.randn(old_mean.numel(), group_size, device=device)).clamp(-1.0, 1.0)
                rewards = score_model(actions.reshape(-1, 1)).reshape_as(actions).square()
                if bias_controlled:
                    rewards = rewards - float(settings["bias_penalty"]) * (actions - batch_truth[:, None]).square()
                advantages = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1.0e-6)
                old_log_prob = -0.5 * ((actions - old_mean[:, None]) / sigma).square()

            for _ in range(int(settings["policy_updates"])):
                mean = policy(batch_features)
                log_prob = -0.5 * ((actions - mean[:, None]) / sigma).square()
                ratio = torch.exp(log_prob - old_log_prob)
                surrogate = torch.minimum(ratio * advantages, ratio.clamp(1.0 - clip, 1.0 + clip) * advantages)
                anchor = torch.mean((mean - baseline_mean).square())
                loss = -surrogate.mean() + float(settings["anchor_strength"]) * anchor
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), float(settings["grad_clip_norm"]))
                optimizer.step()
            epoch_reward += rewards.mean().item()
            batches += 1
        progress.set_postfix(reward=f"{epoch_reward / batches:.5f}")
    return policy.eval()
