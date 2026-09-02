from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from .core import ScoreModel, make_generator, truth_score
from .flow import ConditionalFlow
from .ztautau import candidate_reconstruction, generate_events


def slice_events(events: dict[str, torch.Tensor], indices: torch.Tensor | slice) -> dict[str, torch.Tensor]:
    return {key: value[indices] for key, value in events.items()}


def replacement_information(
    total_reference: torch.Tensor,
    event_reference: torch.Tensor,
    candidate: torch.Tensor,
) -> torch.Tensor:
    return total_reference - event_reference[:, None] + candidate


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
    model, reference, _ = rebuild_policy_score(
        reference_flow, events, config, device,
        int(config["seed"]) + 20000, int(config["training"]["score_epochs"]),
        "frozen reference score",
    )
    return model, reference


def rebuild_policy_score(
    policy: ConditionalFlow,
    events: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    epochs: int,
    description: str,
) -> tuple[ScoreModel, dict[str, torch.Tensor], list[float]]:
    torch.manual_seed(seed + 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 1)
    generator = make_generator(device, seed)
    reference = reconstruct_policy(policy, events, config, generator)
    target = truth_score(events["x"], float(config["physics"]["nominal_C"]))
    valid = reference["valid"]
    settings = config["training"]
    model = ScoreModel(int(settings["score_hidden_width"]), int(settings["score_hidden_layers"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    batch_size = int(settings["batch_size"])
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        raise RuntimeError("The policy produced no physical reconstruction on the score split")
    losses: list[float] = []
    progress = tqdm(range(epochs), desc=description, unit="epoch")
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
        mean_loss = total / permutation.numel()
        losses.append(mean_loss)
        progress.set_postfix(mse=f"{mean_loss:.5f}", valid=f"{valid.float().mean().item():.3f}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    reference["score"] = model(reference["y"]) * valid
    reference["information"] = reference["score"].square()
    return model, reference, losses


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
                replaced_information = replacement_information(
                    information_reference, reference["information"][index], candidate_information,
                )
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


def train_local_dgpo_round(
    current_reference: ConditionalFlow,
    baseline_reference: ConditionalFlow,
    score_model: ScoreModel,
    events: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
    name: str,
    round_index: int,
    seed: int,
    beta_local_override: float | None = None,
    beta_global_override: float | None = None,
) -> tuple[ConditionalFlow, list[dict[str, float]]]:
    """Run one short DGPO step with the unchanged replacement reward."""
    policy = copy.deepcopy(current_reference).to(device).train()
    start_difference = max(
        float((policy_value - reference_value).abs().max())
        for policy_value, reference_value in zip(policy.state_dict().values(), current_reference.state_dict().values())
    )
    for reference_flow in (current_reference, baseline_reference):
        reference_flow.eval()
        for parameter in reference_flow.parameters():
            parameter.requires_grad_(False)
    for parameter in policy.parameters():
        parameter.requires_grad_(True)

    refresh = config["refresh"]
    settings = config["dgpo"]
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]),
    )
    batch_size = int(config["training"]["batch_size"])
    group_size = int(settings["group_size"])
    information_reference = reference["information"].sum().detach()
    beta_local = float(refresh["beta_local"] if beta_local_override is None else beta_local_override)
    beta_global = float(refresh["beta_global"] if beta_global_override is None else beta_global_override)
    epochs = int(refresh["dgpo_epochs_per_round"])
    history: list[dict[str, float]] = []
    progress = tqdm(range(epochs), desc=f"{name} round {round_index + 1}", unit="epoch")
    generator = make_generator(device, seed)
    for local_epoch in progress:
        permutation = torch.randperm(events["x"].numel(), device=device)
        reward_total = 0.0
        local_kl_total = 0.0
        global_kl_total = 0.0
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
                replaced_information = replacement_information(
                    information_reference, reference["information"][index], candidate_information,
                )
                reward = 0.5 * torch.log(replaced_information / information_reference) * float(settings["reward_scale"])
                advantage = reward - (reward.sum(dim=1, keepdim=True) - reward) / (group_size - 1)
            expanded_context = batch["context"][:, None, :].expand(-1, group_size, -1)
            policy_log_prob = policy.log_prob(actions.detach(), expanded_context)
            local_log_prob = current_reference.log_prob(actions, expanded_context)
            global_log_prob = baseline_reference.log_prob(actions, expanded_context)
            local_kl = (sampled_log_prob - local_log_prob).mean()
            global_kl = (sampled_log_prob - global_log_prob).mean()
            loss = -(advantage.detach() * policy_log_prob).mean() + beta_local * local_kl + beta_global * global_kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(config["training"]["grad_clip_norm"]))
            optimizer.step()
            batch_count = index.numel()
            reward_total += reward.mean().item() * batch_count
            local_kl_total += local_kl.item() * batch_count
            global_kl_total += global_kl.item() * batch_count
            invalid_total += (~reconstruction["valid"]).float().mean().item() * batch_count
            seen += batch_count

        metrics = _monitor_policy(
            policy.eval(), current_reference, score_model, events, config,
            seed + 1000 + local_epoch,
        )
        with torch.no_grad():
            monitor_count = min(int(settings["monitor_events"]), events["x"].numel())
            monitor = slice_events(events, slice(0, monitor_count))
            sampled = reconstruct_policy(policy, monitor, config, make_generator(device, seed + 2000 + local_epoch))
            global_monitor_kl = float((
                policy.log_prob(sampled["action"], monitor["context"])
                - baseline_reference.log_prob(sampled["action"], monitor["context"])
            ).mean())
        metrics.update({
            "round": float(round_index),
            "local_epoch": float(local_epoch + 1),
            "epoch": float(round_index * epochs + local_epoch + 1),
            "reward": reward_total / seen,
            "local_kl": local_kl_total / seen,
            "global_kl": global_kl_total / seen,
            "global_kl_monitor": global_monitor_kl,
            "training_invalid_fraction": invalid_total / seen,
            "reference_fisher": float(information_reference),
            "start_max_parameter_difference": start_difference,
        })
        history.append(metrics)
        policy.train()
        progress.set_postfix(reward=f"{metrics['reward']:.4g}", local_kl=f"{metrics['local_kl']:.4g}")
    return policy.eval(), history


def train_iterative_refresh(
    baseline_reference: ConditionalFlow,
    training_events: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
    name: str,
    checkpoints: Path,
    beta_local_override: float | None = None,
    beta_global_override: float | None = None,
) -> tuple[ConditionalFlow, ScoreModel, list[dict[str, float]], list[dict[str, Any]]]:
    refresh = config["refresh"]
    nominal_C = float(config["physics"]["nominal_C"])
    current = copy.deepcopy(baseline_reference).eval()
    history: list[dict[str, float]] = []
    rounds: list[dict[str, Any]] = []
    round_dir = checkpoints / name
    round_dir.mkdir(parents=True, exist_ok=True)
    active_score: ScoreModel | None = None

    for round_index in range(int(refresh["rounds"])):
        score_events = generate_events(
            int(refresh["score_events"]), nominal_C, config, device,
            make_generator(device, int(config["seed"]) + 200000 + 10000 * round_index),
        )
        active_score, _, score_losses = rebuild_policy_score(
            current, score_events, config, device,
            int(config["seed"]) + 210000 + 10000 * round_index,
            int(refresh["score_epochs"]), f"{name} score round {round_index + 1}",
        )
        fisher_events = generate_events(
            int(refresh["fisher_events"]), nominal_C, config, device,
            make_generator(device, int(config["seed"]) + 220000 + 10000 * round_index),
        )
        fisher_reconstruction = make_reference_reconstruction(
            current, active_score, fisher_events, config,
            int(config["seed"]) + 230000 + 10000 * round_index,
        )
        fisher = float(fisher_reconstruction["information"].sum())
        fisher_per_event = float(fisher_reconstruction["information"].mean())
        round_reference = make_reference_reconstruction(
            current, active_score, training_events, config,
            int(config["seed"]) + 240000 + 10000 * round_index,
        )
        next_policy, local_history = train_local_dgpo_round(
            current, baseline_reference, active_score, training_events, round_reference,
            config, device, name, round_index,
            int(config["seed"]) + 250000 + 10000 * round_index,
            beta_local_override, beta_global_override,
        )
        for row in local_history:
            row.update({
                "score_loss": score_losses[-1],
                "round_fisher": fisher,
                "round_fisher_per_event": fisher_per_event,
                "round_predicted_sigma": fisher**-0.5,
                "round_predicted_sigma_per_sqrt_event": fisher_per_event**-0.5,
                "round_valid_fraction": float(fisher_reconstruction["valid"].float().mean()),
            })
        grid = torch.linspace(-1.0, 1.0, int(refresh["score_grid_points"]), device=device)
        with torch.no_grad():
            score_curve = active_score(grid).detach().cpu().numpy()
        torch.save(
            {"method_version": 3, "round": round_index, "state_dict": current.state_dict()},
            round_dir / f"round_{round_index:02d}_reference_policy.pt",
        )
        torch.save(
            {"method_version": 3, "round": round_index, "state_dict": next_policy.state_dict()},
            round_dir / f"round_{round_index:02d}_updated_policy.pt",
        )
        torch.save(
            {"method_version": 3, "round": round_index, "state_dict": active_score.state_dict()},
            round_dir / f"round_{round_index:02d}_score.pt",
        )
        rounds.append({
            "round": round_index,
            "score_loss": score_losses[-1],
            "fisher": fisher,
            "fisher_per_event": fisher_per_event,
            "predicted_sigma": fisher**-0.5,
            "predicted_sigma_per_sqrt_event": fisher_per_event**-0.5,
            "valid_fraction": float(fisher_reconstruction["valid"].float().mean()),
            "score_sample_seed": int(config["seed"]) + 200000 + 10000 * round_index,
            "fisher_sample_seed": int(config["seed"]) + 220000 + 10000 * round_index,
            "training_reference_seed": int(config["seed"]) + 240000 + 10000 * round_index,
            "score_grid": grid.detach().cpu().numpy(),
            "score_curve": score_curve,
        })
        history.extend(local_history)
        current = next_policy

    if active_score is None:
        raise RuntimeError("refresh.rounds must be positive")
    return current, active_score, history, rounds
