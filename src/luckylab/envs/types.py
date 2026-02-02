"""Type definitions for luckylab environments."""

import torch

# Observation type: dict of observation tensors
VecEnvObs = dict[str, torch.Tensor | dict[str, torch.Tensor]]

# Step return type: (obs, reward, terminated, truncated, info)
VecEnvStepReturn = tuple[VecEnvObs, torch.Tensor, torch.Tensor, torch.Tensor, dict]
