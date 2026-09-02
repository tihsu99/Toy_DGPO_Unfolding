from __future__ import annotations

from typing import Any

import torch


def _normalize(vector: torch.Tensor, epsilon: float = 1.0e-12) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(epsilon)


def minkowski_mass2(momentum: torch.Tensor) -> torch.Tensor:
    return momentum[..., 0].square() - momentum[..., 1:].square().sum(dim=-1)


def boost(momentum: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    beta2 = beta.square().sum(dim=-1).clamp_max(1.0 - 1.0e-12)
    gamma = torch.rsqrt(1.0 - beta2)
    beta_dot_p = (beta * momentum[..., 1:]).sum(dim=-1)
    factor = torch.where(beta2 > 1.0e-14, (gamma - 1.0) * beta_dot_p / beta2 + gamma * momentum[..., 0], gamma * momentum[..., 0])
    energy = gamma * (momentum[..., 0] + beta_dot_p)
    spatial = momentum[..., 1:] + factor[..., None] * beta
    return torch.cat((energy[..., None], spatial), dim=-1)


def tangent_basis(direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z_axis = torch.zeros_like(direction)
    z_axis[..., 2] = 1.0
    x_axis = torch.zeros_like(direction)
    x_axis[..., 0] = 1.0
    reference = torch.where((direction[..., 2].abs() < 0.9)[..., None], z_axis, x_axis)
    e1 = _normalize(torch.linalg.cross(reference, direction, dim=-1))
    e2 = _normalize(torch.linalg.cross(direction, e1, dim=-1))
    return e1, e2


def sphere_log_map(seed: torch.Tensor, target: torch.Tensor, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
    cosine = (seed * target).sum(dim=-1).clamp(-1.0, 1.0)
    gamma = torch.acos(cosine)
    tangent = target - cosine[..., None] * seed
    unit = _normalize(tangent)
    coefficients = torch.stack(((unit * e1).sum(dim=-1), (unit * e2).sum(dim=-1)), dim=-1)
    return torch.where((gamma > 1.0e-9)[..., None], gamma[..., None] * coefficients, torch.zeros_like(coefficients))


def sphere_exp_map(seed: torch.Tensor, action: torch.Tensor, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
    rho = torch.linalg.vector_norm(action, dim=-1)
    tangent = action[..., 0, None] * e1 + action[..., 1, None] * e2
    sinc = torch.where(rho > 1.0e-9, torch.sin(rho) / rho, 1.0 - rho.square() / 6.0)
    return _normalize(torch.cos(rho)[..., None] * seed + sinc[..., None] * tangent)


def _sample_linear(coefficient: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    u = torch.rand(coefficient.shape, device=coefficient.device, dtype=coefficient.dtype, generator=generator)
    uniform = 2.0 * u - 1.0
    safe = coefficient.abs() > 1.0e-8
    denominator = torch.where(safe, coefficient, torch.ones_like(coefficient))
    value = (-1.0 + torch.sqrt((1.0 - coefficient) ** 2 + 4.0 * coefficient * u)) / denominator
    return torch.where(safe, value, uniform)


def sample_spin_angles(
    count: int, C: float, device: torch.device, generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    if abs(C) >= 1.0:
        raise ValueError("C must satisfy |C| < 1")
    c_a = 2.0 * torch.rand(count, device=device, dtype=dtype, generator=generator) - 1.0
    c_b = _sample_linear(C * c_a, generator)
    return c_a, c_b


def _sample_tau_direction(
    count: int, device: torch.device, generator: torch.Generator, tolerance: float, dtype: torch.dtype,
) -> torch.Tensor:
    cosine = 2.0 * torch.rand(count, device=device, dtype=dtype, generator=generator) - 1.0
    singular = 1.0 - cosine.square() <= tolerance**2
    while torch.any(singular):
        cosine[singular] = 2.0 * torch.rand(int(singular.sum()), device=device, dtype=dtype, generator=generator) - 1.0
        singular = 1.0 - cosine.square() <= tolerance**2
    phi = 2.0 * torch.pi * torch.rand(count, device=device, dtype=dtype, generator=generator)
    sine = torch.sqrt(1.0 - cosine.square())
    return torch.stack((sine * torch.cos(phi), sine * torch.sin(phi), cosine), dim=-1)


def _visible_direction(cosine_n: torch.Tensor, n_axis: torch.Tensor, tau_axis: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    phi = 2.0 * torch.pi * torch.rand(cosine_n.shape, device=cosine_n.device, dtype=cosine_n.dtype, generator=generator)
    transverse = _normalize(torch.linalg.cross(n_axis, tau_axis, dim=-1))
    radius = torch.sqrt((1.0 - cosine_n.square()).clamp_min(0.0))
    return _normalize(cosine_n[..., None] * n_axis + radius[..., None] * (torch.cos(phi)[..., None] * tau_axis + torch.sin(phi)[..., None] * transverse))


def _sample_beta22(count: int, device: torch.device, generator: torch.Generator, dtype: torch.dtype) -> torch.Tensor:
    accepted = torch.empty(0, device=device, dtype=dtype)
    while accepted.numel() < count:
        proposal = torch.rand(max(count, 2 * (count - accepted.numel())), device=device, dtype=dtype, generator=generator)
        height = 1.5 * torch.rand(proposal.shape, device=device, dtype=dtype, generator=generator)
        accepted = torch.cat((accepted, proposal[height <= 6.0 * proposal * (1.0 - proposal)]))
    return accepted[:count]


def _smear_visible(momentum: torch.Tensor, detector: dict[str, Any], generator: torch.Generator) -> torch.Tensor:
    energy = momentum[:, 0]
    direction = _normalize(momentum[:, 1:])
    e1, e2 = tangent_basis(direction)
    angular_sigma = float(detector["angular_resolution"])
    angular_action = angular_sigma * torch.randn((energy.numel(), 2), device=energy.device, dtype=energy.dtype, generator=generator)
    smeared_direction = sphere_exp_map(direction, angular_action, e1, e2)
    scale = (1.0 + float(detector["relative_energy_resolution"]) * torch.randn(energy.shape, device=energy.device, dtype=energy.dtype, generator=generator)).clamp_min(1.0e-3)
    smeared_energy = energy * scale
    return torch.cat((smeared_energy[:, None], smeared_energy[:, None] * smeared_direction), dim=-1)


def observed_context(visible_a: torch.Tensor, visible_b: torch.Tensor, energy_tau: float) -> torch.Tensor:
    return torch.cat((visible_a, visible_b), dim=-1) / energy_tau


def generate_events(count: int, C: float, config: dict[str, Any], device: torch.device, generator: torch.Generator) -> dict[str, torch.Tensor]:
    physics = config["physics"]
    m_z = float(physics["m_Z"])
    m_tau = float(physics["m_tau"])
    energy_tau = 0.5 * m_z
    momentum_tau = (energy_tau**2 - m_tau**2) ** 0.5
    physics_dtype = torch.float32 if device.type == "mps" else torch.float64
    c_a, c_b = sample_spin_angles(count, C, device, generator, physics_dtype)
    x = c_a * c_b
    k_a = _sample_tau_direction(count, device, generator, float(physics["axis_singularity_tolerance"]), physics_dtype)
    k_b = -k_a
    beam = torch.zeros_like(k_a)
    beam[:, 2] = 1.0
    n_axis = _normalize(torch.linalg.cross(beam, k_a, dim=-1))
    direction_a_star = _visible_direction(c_a, n_axis, k_a, generator)
    direction_b_star = _visible_direction(c_b, n_axis, k_b, generator)
    if float(physics["visible_beta_alpha"]) != 2.0 or float(physics["visible_beta_beta"]) != 2.0:
        raise ValueError("The implemented minimal decay uses the specified Beta(2, 2) visible fraction")
    r_a = _sample_beta22(count, device, generator, physics_dtype)
    r_b = _sample_beta22(count, device, generator, physics_dtype)
    energy_a_star = 0.5 * m_tau * r_a
    energy_b_star = 0.5 * m_tau * r_b
    visible_a_star = torch.cat((energy_a_star[:, None], energy_a_star[:, None] * direction_a_star), dim=-1)
    visible_b_star = torch.cat((energy_b_star[:, None], energy_b_star[:, None] * direction_b_star), dim=-1)
    tau_rest = torch.zeros((count, 4), device=device, dtype=physics_dtype)
    tau_rest[:, 0] = m_tau
    invisible_a_star = tau_rest - visible_a_star
    invisible_b_star = tau_rest - visible_b_star
    beta_a = (momentum_tau / energy_tau) * k_a
    beta_b = (momentum_tau / energy_tau) * k_b
    visible_a = boost(visible_a_star, beta_a)
    visible_b = boost(visible_b_star, beta_b)
    invisible_a = boost(invisible_a_star, beta_a)
    invisible_b = boost(invisible_b_star, beta_b)
    tau_a = torch.cat((torch.full((count, 1), energy_tau, device=device, dtype=physics_dtype), momentum_tau * k_a), dim=-1)
    tau_b = torch.cat((torch.full((count, 1), energy_tau, device=device, dtype=physics_dtype), momentum_tau * k_b), dim=-1)
    closure = torch.maximum((tau_a - visible_a - invisible_a).abs().amax(dim=-1), (tau_b - visible_b - invisible_b).abs().amax(dim=-1))
    visible_a_obs = _smear_visible(visible_a, config["detector"], generator)
    visible_b_obs = _smear_visible(visible_b, config["detector"], generator)
    direction_a_obs = _normalize(visible_a_obs[:, 1:])
    direction_b_obs = _normalize(visible_b_obs[:, 1:])
    seed = _normalize(direction_a_obs - direction_b_obs)
    seed_e1, seed_e2 = tangent_basis(seed)
    target_action = sphere_log_map(seed, k_a, seed_e1, seed_e2)
    generated = {
        "c_a": c_a, "c_b": c_b, "x": x, "k_true": k_a, "r_a": r_a, "r_b": r_b,
        "invisible_mass_a": torch.sqrt(minkowski_mass2(invisible_a_star).clamp_min(0.0)),
        "invisible_mass_b": torch.sqrt(minkowski_mass2(invisible_b_star).clamp_min(0.0)),
        "visible_true_a": visible_a, "visible_true_b": visible_b,
        "visible_obs_a": visible_a_obs, "visible_obs_b": visible_b_obs,
        "closure": closure, "seed": seed, "seed_e1": seed_e1, "seed_e2": seed_e2,
        "target_action": target_action, "context": observed_context(visible_a_obs, visible_b_obs, energy_tau),
    }
    return {key: value if key == "closure" else value.float() for key, value in generated.items()}


def candidate_reconstruction(events: dict[str, torch.Tensor], actions: torch.Tensor, config: dict[str, Any]) -> dict[str, torch.Tensor]:
    squeeze = actions.ndim == 2
    if squeeze:
        actions = actions[:, None, :]
    batch, candidates, _ = actions.shape
    seed = events["seed"][:, None, :].expand(-1, candidates, -1)
    e1 = events["seed_e1"][:, None, :].expand_as(seed)
    e2 = events["seed_e2"][:, None, :].expand_as(seed)
    k_a = sphere_exp_map(seed, actions, e1, e2)
    k_b = -k_a
    m_z = float(config["physics"]["m_Z"])
    m_tau = float(config["physics"]["m_tau"])
    energy_tau = 0.5 * m_z
    momentum_tau = (energy_tau**2 - m_tau**2) ** 0.5
    tau_a = torch.cat((torch.full((batch, candidates, 1), energy_tau, device=actions.device), momentum_tau * k_a), dim=-1)
    tau_b = torch.cat((torch.full((batch, candidates, 1), energy_tau, device=actions.device), momentum_tau * k_b), dim=-1)
    visible_a = events["visible_obs_a"][:, None, :].expand(-1, candidates, -1)
    visible_b = events["visible_obs_b"][:, None, :].expand(-1, candidates, -1)
    invisible_a = tau_a - visible_a
    invisible_b = tau_b - visible_b
    tolerance = float(config["physics"]["validity_tolerance"])
    valid = (invisible_a[..., 0] > 0.0) & (invisible_b[..., 0] > 0.0)
    valid &= (minkowski_mass2(invisible_a) >= -tolerance) & (minkowski_mass2(invisible_b) >= -tolerance)
    beta_a = (momentum_tau / energy_tau) * k_a
    beta_b = (momentum_tau / energy_tau) * k_b
    visible_a_star = boost(visible_a, -beta_a)
    visible_b_star = boost(visible_b, -beta_b)
    beam = torch.zeros_like(k_a)
    beam[..., 2] = 1.0
    normal_raw = torch.linalg.cross(beam, k_a, dim=-1)
    normal_norm = torch.linalg.vector_norm(normal_raw, dim=-1)
    valid &= normal_norm > float(config["physics"]["axis_singularity_tolerance"])
    normal = _normalize(normal_raw)
    direction_a_star = _normalize(visible_a_star[..., 1:])
    direction_b_star = _normalize(visible_b_star[..., 1:])
    c_a = (direction_a_star * normal).sum(dim=-1).clamp(-1.0, 1.0)
    c_b = (direction_b_star * normal).sum(dim=-1).clamp(-1.0, 1.0)
    result = {
        "y": c_a * c_b, "valid": valid, "k_a": k_a,
        "mass2_a": minkowski_mass2(invisible_a), "mass2_b": minkowski_mass2(invisible_b),
    }
    return {key: value[:, 0] for key, value in result.items()} if squeeze else result
