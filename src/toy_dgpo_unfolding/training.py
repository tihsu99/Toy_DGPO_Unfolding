from __future__ import annotations

import copy
from typing import Any

import torch
from tqdm.auto import tqdm

from .core import ScoreModel, make_generator, truth_score
from .flow import ConditionalFlow
from .ztautau import candidate_reconstruction


def slice_events(events: dict[str, torch.Tensor], indices: torch.Tensor | slice) -> dict[str, torch.Tensor]:
    return {key: value[indices] for key, value in events.items()}


def make_flow(config: dict[str, Any], device: torch.device) -> ConditionalFlow:
    settings = config["flow"]
    return ConditionalFlow(
        int(settings["context_dim"]), int(settings["hidden_width"]),
        int(settings["coupling_layers"]), float(settings["scale_limit"]),
    ).to(device)


def train_baseline(events: dict[str, torch.Tensor], config: dict[str, Any], device: torch.device) -> ConditionalFlow:
    settings = config["training"]
    flow = make_flow(config, device)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    batch_size = int(settings["batch_size"])
    progress = tqdm(range(int(settings["baseline_epochs"])), desc="baseline flow", unit="epoch")
    for _ in progress:
        permutation = torch.randperm(events["x"].numel(), device=device)
        total = 0.0
        for start in range(0, permutation.numel(), batch_size):
            index = permutation[start : start + batch_size]
            loss = -flow.log_prob(events["target_action"][index], events["context"][index]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), float(settings["grad_clip_norm"]))
            optimizer.step()
            total += loss.item() * index.numel()
        progress.set_postfix(nll=f"{total / permutation.numel():.5f}")
    return flow.eval()


@torch.no_grad()
def reconstruct_policy(
    flow: ConditionalFlow,
    events: dict[str, torch.Tensor],
    config: dict[str, Any],
    generator: torch.Generator,
    candidates: int = 1,
) -> dict[str, torch.Tensor]:
    batch_size = int(config["training"]["batch_size"])
    pieces: dict[str, list[torch.Tensor]] = {"action": [], "log_prob": [], "y": [], "valid": [], "k_a": []}
    for start in range(0, events["x"].numel(), batch_size):
        batch = slice_events(events, slice(start, start + batch_size))
        action, log_prob = flow.sample(batch["context"], candidates, generator)
        reconstruction = candidate_reconstruction(batch, action, config)
        pieces["action"].append(action)
        pieces["log_prob"].append(log_prob)
        pieces["y"].append(reconstruction["y"])
        pieces["valid"].append(reconstruction["valid"])
        pieces["k_a"].append(reconstruction["k_a"])
    result = {key: torch.cat(value) for key, value in pieces.items()}
    if candidates == 1:
        result = {key: value[:, 0] for key, value in result.items()}
    return result


def train_reference_score(
    reference_flow: ConditionalFlow,
    events: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[ScoreModel, dict[str, torch.Tensor]]:
    generator = make_generator(device, int(config["seed"]) + 20000)
    reference = reconstruct_policy(reference_flow, events, config, generator)
    target = truth_score(events["x"], float(config["physics"]["nominal_C"]))
    valid = reference["valid"]
    settings = config["training"]
    model = ScoreModel(int(settings["score_hidden_width"]), int(settings["score_hidden_layers"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    batch_size = int(settings["batch_size"])
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        raise RuntimeError("The baseline produced no physical reconstruction on the score split")
    progress = tqdm(range(int(settings["score_epochs"])), desc="frozen reference score", unit="epoch")
    for _ in progress:
        permutation = valid_indices[torch.randperm(valid_indices.numel(), device=device)]
        total = 0.0
        for start in range(0, permutation.numel(), batch_size):
            index = permutation[start : start + batch_size]
            loss = (model(reference["y"][index]) - target[index]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * index.numel()
        progress.set_postfix(mse=f"{total / permutation.numel():.5f}", valid=f"{valid.float().mean().item():.3f}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    reference["score"] = model(reference["y"]) * valid
    reference["information"] = reference["score"].square()
    return model, reference


@torch.no_grad()
def make_reference_reconstruction(
    reference_flow: ConditionalFlow,
    score_model: ScoreModel,
    events: dict[str, torch.Tensor],
    config: dict[str, Any],
    seed: int,
) -> dict[str, torch.Tensor]:
    reference = reconstruct_policy(reference_flow, events, config, make_generator(events["x"].device, seed))
    reference["score"] = score_model(reference["y"]) * reference["valid"]
    reference["information"] = reference["score"].square()
    return reference


def _monitor_policy(
    policy: ConditionalFlow,
    reference_flow: ConditionalFlow,
    score_model: ScoreModel,
    events: dict[str, torch.Tensor],
    config: dict[str, Any],
    seed: int,
) -> dict[str, float]:
    count = min(int(config["dgpo"]["monitor_events"]), events["x"].numel())
    monitor = slice_events(events, slice(0, count))
    generator = make_generator(events["x"].device, seed)
    reconstructed = reconstruct_policy(policy, monitor, config, generator)
    score = score_model(reconstructed["y"]) * reconstructed["valid"]
    information = float(score.square().sum())
    score_sum = float(score.sum())
    with torch.no_grad():
        action = reconstructed["action"]
        kl = float((policy.log_prob(action, monitor["context"]) - reference_flow.log_prob(action, monitor["context"])).mean())
        cosine = (reconstructed["k_a"] * monitor["k_true"]).sum(dim=-1).clamp(-1.0, 1.0)
        angular_error = float(torch.acos(cosine).mean())
    return {
        "fisher": information,
        "predicted_sigma": information**-0.5 if information > 0.0 else float("inf"),
        "score_balance": score_sum / information if information > 0.0 else float("nan"),
        "bias_significance": abs(score_sum) / information**0.5 if information > 0.0 else float("nan"),
        "kl_to_reference": kl,
        "invalid_fraction": 1.0 - float(reconstructed["valid"].float().mean()),
        "angular_error": angular_error,
    }


def train_dgpo(
    reference_flow: ConditionalFlow,
    score_model: ScoreModel,
    events: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
    name: str,
    kl_coefficient: float,
) -> tuple[ConditionalFlow, list[dict[str, float]]]:
    policy = copy.deepcopy(reference_flow).to(device).train()
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    reference_flow.eval()
    for parameter in reference_flow.parameters():
        parameter.requires_grad_(False)
    settings = config["dgpo"]
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    batch_size = int(config["training"]["batch_size"])
    group_size = int(settings["group_size"])
    information_reference = reference["information"].sum().detach()
    history: list[dict[str, float]] = []
    progress = tqdm(range(int(settings["epochs"])), desc=name.replace("_", " "), unit="epoch")
    generator = make_generator(device, int(config["seed"]) + 30000 + (0 if kl_coefficient == 0.0 else 1000))
    for epoch in progress:
        permutation = torch.randperm(events["x"].numel(), device=device)
        reward_total = 0.0
        invalid_total = 0.0
        seen = 0
        for start in range(0, permutation.numel(), batch_size):
            index = permutation[start : start + batch_size]
            batch = slice_events(events, index)
            actions, sampled_log_prob = policy.sample(batch["context"], group_size, generator)
            with torch.no_grad():
                reconstruction = candidate_reconstruction(batch, actions.detach(), config)
                candidate_score = score_model(reconstruction["y"]) * reconstruction["valid"]
                candidate_information = candidate_score.square()
                replaced_information = information_reference - reference["information"][index, None] + candidate_information
                reward = 0.5 * torch.log(replaced_information / information_reference) * float(settings["reward_scale"])
                advantage = reward - (reward.sum(dim=1, keepdim=True) - reward) / (group_size - 1)
            expanded_context = batch["context"][:, None, :].expand(-1, group_size, -1)
            policy_log_prob = policy.log_prob(actions.detach(), expanded_context)
            policy_gradient = -(advantage.detach() * policy_log_prob).mean()
            reference_log_prob = reference_flow.log_prob(actions, expanded_context)
            kl = (sampled_log_prob - reference_log_prob).mean()
            loss = policy_gradient + kl_coefficient * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(config["training"]["grad_clip_norm"]))
            optimizer.step()
            reward_total += reward.mean().item() * index.numel()
            invalid_total += (~reconstruction["valid"]).float().mean().item() * index.numel()
            seen += index.numel()
        metrics = _monitor_policy(policy.eval(), reference_flow, score_model, events, config, int(config["seed"]) + 40000 + epoch)
        metrics.update({"epoch": float(epoch + 1), "reward": reward_total / seen, "training_invalid_fraction": invalid_total / seen})
        history.append(metrics)
        policy.train()
        progress.set_postfix(reward=f"{metrics['reward']:.4g}", kl=f"{metrics['kl_to_reference']:.4g}", invalid=f"{metrics['invalid_fraction']:.3f}")
    return policy.eval(), history
