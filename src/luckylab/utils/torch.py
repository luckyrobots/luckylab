"""PyTorch backend configuration utilities."""

from __future__ import annotations

import torch
from packaging.version import parse


def configure_torch_backends(
    allow_tf32: bool = True,
    deterministic: bool = False,
) -> None:
    """Configure PyTorch backends for performance/reproducibility and NaN debugging.

    Args:
        allow_tf32: If True, use TF32 precision for faster computation on Ampere+ GPUs.
            If False, use standard IEEE FP32 precision. Only applies to CUDA.
        deterministic: If True, use deterministic algorithms (slower but reproducible).
            If False, allow cuDNN to benchmark and select fastest algorithms. Only
            applies to CUDA.

    Note:
        TF32 uses reduced precision (10-bit mantissa vs 23-bit for FP32) for internal
        matrix multiplications providing a speedup with minimal impact on accuracy.

        See https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere for details.

    Example:
        >>> from luckylab.utils import configure_torch_backends
        >>> configure_torch_backends(device="cuda", allow_tf32=True)
    """
    torch_version = parse(torch.__version__.split("+")[0])
    if torch_version >= parse("2.9.0"):
        _configure_29(allow_tf32)
    else:
        _configure_pre29(allow_tf32)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def _configure_29(allow_tf32: bool) -> None:
    """Configure PyTorch CUDA and cuDNN backends for PyTorch 2.9+."""
    # tf32 for performance, ieee for full FP32 accuracy.
    precision = "tf32" if allow_tf32 else "ieee"
    torch.backends.cuda.matmul.fp32_precision = precision
    torch.backends.cudnn.fp32_precision = precision  # type: ignore


def _configure_pre29(allow_tf32: bool) -> None:
    """Configure PyTorch CUDA and cuDNN backends for PyTorch <2.9."""
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
