import torch

VecEnvObs = dict[str, torch.Tensor | dict[str, torch.Tensor]]
VecEnvStepReturn = tuple[VecEnvObs, torch.Tensor, torch.Tensor, torch.Tensor, dict]