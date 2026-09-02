from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .core import bounded_to_latent, clone_model, make_policy, make_score_model, nominal_score, sample_policy


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
        mean = baseline(features.to(device))
        reco = sample_policy(mean, sigma, torch.randn(mean.shape, device=device))
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
    baseline_score_model: nn.Module,
    features: torch.Tensor,
    truth: torch.Tensor,
    score_features: torch.Tensor,
    score_truth: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    bias_controlled: bool,
) -> tuple[nn.Module, nn.Module]:
    settings = config["training"]
    policy = clone_model(baseline).to(device).train()
    frozen_baseline = clone_model(baseline).to(device).eval()
    score_model = clone_model(baseline_score_model).to(device)
    for parameter in frozen_baseline.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"])
    )
    loader = _loader(features, truth, batch_size=int(settings["batch_size"]))
    score_loader = _loader(score_features, score_truth, batch_size=int(settings["batch_size"]))
    score_optimizer = torch.optim.AdamW(
        score_model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"])
    )
    group_size = int(settings["group_size"])
    sigma = float(config["model"]["policy_sigma"])
    clip = float(settings["clip_epsilon"])
    label = "bias-controlled DGPO" if bias_controlled else "Fisher DGPO"
    progress = tqdm(range(int(settings["dgpo_epochs"])), desc=label, unit="epoch")
    calibration_edges = torch.linspace(
        -1.0, 1.0, int(settings["conditional_calibration_bins"]) + 1, device=device
    )
    calibration_strength = float(settings["conditional_calibration_strength"])
    if bias_controlled:
        calibration_strength *= float(settings["bias_controlled_calibration_multiplier"])
    nominal_C = float(config["physics"]["nominal_C"])
    for epoch in progress:
        score_mse = 0.0
        score_seen = 0
        if epoch % int(settings["score_refresh_interval"]) == 0:
            score_model.train()
            for parameter in score_model.parameters():
                parameter.requires_grad_(True)
            for _ in range(int(settings["score_refresh_epochs"])):
                for batch_features, batch_truth in score_loader:
                    batch_features, batch_truth = batch_features.to(device), batch_truth.to(device)
                    with torch.no_grad():
                        current_mean = policy(batch_features)
                        current_reco = sample_policy(
                            current_mean, sigma, torch.randn(current_mean.shape, device=device)
                        )
                        targets = nominal_score(batch_truth, nominal_C)
                    score_loss = torch.mean((score_model(current_reco[:, None]) - targets).square())
                    score_optimizer.zero_grad(set_to_none=True)
                    score_loss.backward()
                    score_optimizer.step()
                    score_mse += score_loss.item() * batch_truth.numel()
                    score_seen += batch_truth.numel()
            score_model.eval()
            for parameter in score_model.parameters():
                parameter.requires_grad_(False)

        epoch_reward = 0.0
        epoch_calibration = 0.0
        batches = 0
        for batch_features, batch_truth in loader:
            batch_features, batch_truth = batch_features.to(device), batch_truth.to(device)
            with torch.no_grad():
                old_mean = policy(batch_features)
                baseline_mean = frozen_baseline(batch_features)
                old_latent_mean = bounded_to_latent(old_mean)
                latent_actions = old_latent_mean[:, None] + sigma * torch.randn(
                    old_mean.numel(), group_size, device=device
                )
                actions = torch.tanh(latent_actions)
                rewards = score_model(actions.reshape(-1, 1)).reshape_as(actions).square()
                if bias_controlled:
                    rewards = rewards - float(settings["bias_penalty"]) * (actions - batch_truth[:, None]).square()
                advantages = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1.0e-6)
                old_log_prob = -0.5 * ((latent_actions - old_latent_mean[:, None]) / sigma).square()

            for _ in range(int(settings["policy_updates"])):
                mean = policy(batch_features)
                latent_mean = bounded_to_latent(mean)
                log_prob = -0.5 * ((latent_actions - latent_mean[:, None]) / sigma).square()
                ratio = torch.exp(log_prob - old_log_prob)
                surrogate = torch.minimum(ratio * advantages, ratio.clamp(1.0 - clip, 1.0 + clip) * advantages)
                anchor = torch.mean((mean - baseline_mean).square())
                truth_bin = torch.bucketize(batch_truth, calibration_edges[1:-1])
                calibration_terms = [
                    (mean[truth_bin == index].mean() - batch_truth[truth_bin == index].mean()).square()
                    for index in range(calibration_edges.numel() - 1)
                    if torch.any(truth_bin == index)
                ]
                calibration = torch.stack(calibration_terms).mean()
                loss = (
                    -surrogate.mean()
                    + float(settings["anchor_strength"]) * anchor
                    + calibration_strength * calibration
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), float(settings["grad_clip_norm"]))
                optimizer.step()
            epoch_reward += rewards.mean().item()
            epoch_calibration += calibration.item()
            batches += 1
        progress.set_postfix(
            reward=f"{epoch_reward / batches:.5f}",
            calibration=f"{epoch_calibration / batches:.5f}",
            score_mse=f"{score_mse / max(score_seen, 1):.5f}",
        )
    return policy.eval(), score_model.eval()
