from __future__ import annotations

import math

import torch
from torch import nn


class ConditionalFlow(nn.Module):
    def __init__(self, context_dim: int, width: int, layers: int, scale_limit: float):
        super().__init__()
        self.scale_limit = scale_limit
        self.mask_names: list[str] = []
        self.conditioners = nn.ModuleList()
        for index in range(layers):
            name = f"mask_{index}"
            self.register_buffer(name, torch.tensor([1.0, 0.0] if index % 2 == 0 else [0.0, 1.0]))
            self.mask_names.append(name)
            self.conditioners.append(nn.Sequential(nn.Linear(context_dim + 2, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 4)))

    def _transform(self, value: torch.Tensor, context: torch.Tensor, inverse: bool) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(value.shape[:-1], device=value.device)
        indices = range(len(self.conditioners) - 1, -1, -1) if inverse else range(len(self.conditioners))
        for index in indices:
            mask = getattr(self, self.mask_names[index])
            fixed = value * mask
            shift, raw_scale = self.conditioners[index](torch.cat((fixed, context), dim=-1)).chunk(2, dim=-1)
            scale = self.scale_limit * torch.tanh(raw_scale) * (1.0 - mask)
            shift = shift * (1.0 - mask)
            if inverse:
                value = fixed + (1.0 - mask) * (value - shift) * torch.exp(-scale)
                log_det -= scale.sum(dim=-1)
            else:
                value = fixed + (1.0 - mask) * (value * torch.exp(scale) + shift)
                log_det += scale.sum(dim=-1)
        return value, log_det

    def log_prob(self, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        base, log_det = self._transform(action, context, inverse=True)
        return -0.5 * (base.square() + math.log(2.0 * math.pi)).sum(dim=-1) + log_det

    def sample(self, context: torch.Tensor, count: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        base = torch.randn((context.shape[0], count, 2), device=context.device, generator=generator)
        expanded_context = context[:, None, :].expand(-1, count, -1)
        action, forward_log_det = self._transform(base, expanded_context, inverse=False)
        log_prob = -0.5 * (base.square() + math.log(2.0 * math.pi)).sum(dim=-1) - forward_log_det
        return action, log_prob

