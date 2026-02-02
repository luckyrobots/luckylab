"""Noise configuration classes for domain randomization.

Matches mjlab/utils/noise/noise_cfg.py.
Uses torch tensors for GPU-native operations.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(kw_only=True)
class NoiseCfg(abc.ABC):
    """Base configuration for a noise term."""

    operation: Literal["add", "scale", "abs"] = "add"
    """How to apply noise: 'add' (obs + noise), 'scale' (obs * noise), or 'abs' (replace with noise)."""

    @abc.abstractmethod
    def apply(self, data: torch.Tensor) -> torch.Tensor:
        """Apply noise to the input data.

        Args:
            data: Input tensor to apply noise to.

        Returns:
            Noisy data tensor.
        """
        raise NotImplementedError


@dataclass
class UniformNoiseCfg(NoiseCfg):
    """Uniform noise configuration.

    Matches mjlab's UniformNoiseCfg API.

    Args:
        n_min: Lower bound of the uniform distribution.
        n_max: Upper bound of the uniform distribution.
        operation: How to apply noise ('add', 'scale', or 'abs').

    Example:
        # Additive uniform noise in [-0.1, 0.1]
        noise = UniformNoiseCfg(n_min=-0.1, n_max=0.1)

        # Multiplicative noise (scale by U[0.9, 1.1])
        noise = UniformNoiseCfg(n_min=0.9, n_max=1.1, operation='scale')
    """

    n_min: torch.Tensor | float = -0.1
    n_max: torch.Tensor | float = 0.1

    def __post_init__(self):
        if isinstance(self.n_min, (int, float)) and isinstance(self.n_max, (int, float)):
            if self.n_min >= self.n_max:
                raise ValueError(f"n_min ({self.n_min}) must be less than n_max ({self.n_max})")

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        # Ensure bounds are on correct device
        n_min = self.n_min
        n_max = self.n_max
        if isinstance(n_min, torch.Tensor):
            n_min = n_min.to(device=data.device)
        if isinstance(n_max, torch.Tensor):
            n_max = n_max.to(device=data.device)

        # Generate uniform noise in [0, 1) and scale to [n_min, n_max)
        noise = torch.rand_like(data) * (n_max - n_min) + n_min

        if self.operation == "add":
            return data + noise
        elif self.operation == "scale":
            return data * noise
        elif self.operation == "abs":
            return noise
        else:
            raise ValueError(f"Unsupported noise operation: {self.operation}")


@dataclass
class GaussianNoiseCfg(NoiseCfg):
    """Gaussian (normal) noise configuration.

    Matches mjlab's GaussianNoiseCfg API.

    Args:
        mean: Mean of the Gaussian distribution.
        std: Standard deviation of the Gaussian distribution.
        operation: How to apply noise ('add', 'scale', or 'abs').

    Example:
        # Additive noise with std=0.1
        noise = GaussianNoiseCfg(std=0.1)

        # Multiplicative noise (scale by 1 + N(0, 0.05))
        noise = GaussianNoiseCfg(mean=1.0, std=0.05, operation='scale')
    """

    mean: torch.Tensor | float = 0.0
    std: torch.Tensor | float = 0.1

    def __post_init__(self):
        if isinstance(self.std, (int, float)) and self.std <= 0:
            raise ValueError(f"std ({self.std}) must be positive")

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        # Ensure parameters are on correct device
        mean = self.mean
        std = self.std
        if isinstance(mean, torch.Tensor):
            mean = mean.to(device=data.device)
        if isinstance(std, torch.Tensor):
            std = std.to(device=data.device)

        # Generate standard normal noise and scale
        noise = mean + std * torch.randn_like(data)

        if self.operation == "add":
            return data + noise
        elif self.operation == "scale":
            return data * noise
        elif self.operation == "abs":
            return noise
        else:
            raise ValueError(f"Unsupported noise operation: {self.operation}")
