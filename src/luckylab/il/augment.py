"""Data augmentation transforms for IL training.

All transforms operate on batched image tensors (B, C, H, W) or (B, T, C, H, W).
They are applied between preprocessing and the policy forward pass.
"""

from __future__ import annotations

import io
import random
from functools import wraps
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from luckylab.il.config import IlRunnerCfg


def _flatten_5d(fn):
    """Decorator that handles (B, T, C, H, W) tensors by flattening to (B*T, C, H, W)."""
    @wraps(fn)
    def wrapper(imgs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if imgs.ndim == 5:
            B, T, C, H, W = imgs.shape
            return fn(imgs.reshape(B * T, C, H, W), *args, **kwargs).reshape(B, T, C, H, W)
        return fn(imgs, *args, **kwargs)
    return wrapper


def apply_augmentations(batch: dict, cfg: IlRunnerCfg) -> dict:
    """Apply all enabled augmentations to a training batch in-place."""
    has_visual = cfg.grayscale or cfg.camera_noise or cfg.random_erasing
    has_dynamics = cfg.state_noise_std > 0 or cfg.action_noise_std > 0

    if not has_visual and not has_dynamics:
        return batch

    if has_visual:
        for key in [k for k in batch if k.startswith("observation.images.")]:
            imgs = batch[key]
            if not imgs.is_floating_point():
                imgs = imgs.float()
            if cfg.grayscale:
                imgs = grayscale(imgs)
            if cfg.camera_noise:
                imgs = camera_noise(
                    imgs,
                    noise_std=cfg.camera_noise_std,
                    blur_sigma=cfg.camera_blur_sigma,
                    blur_p=cfg.camera_blur_p,
                    jpeg_quality=cfg.camera_jpeg_quality,
                    jpeg_p=cfg.camera_jpeg_p,
                )
            if cfg.random_erasing:
                imgs = random_erase(imgs, p=cfg.random_erasing_p, scale=cfg.random_erasing_scale)
            batch[key] = imgs

    if cfg.state_noise_std > 0 and "observation.state" in batch:
        s = batch["observation.state"]
        if s.is_floating_point():
            batch["observation.state"] = s + torch.randn_like(s) * cfg.state_noise_std

    if cfg.action_noise_std > 0 and "action" in batch:
        a = batch["action"]
        if a.is_floating_point():
            batch["action"] = a + torch.randn_like(a) * cfg.action_noise_std

    return batch


def log_active(cfg: IlRunnerCfg) -> list[str]:
    """Return human-readable list of active augmentation names."""
    active = []
    if cfg.grayscale:
        active.append("grayscale")
    if cfg.camera_noise:
        active.append(f"camera_noise(std={cfg.camera_noise_std})")
    if cfg.random_erasing:
        active.append(f"random_erasing(p={cfg.random_erasing_p})")
    if cfg.state_noise_std > 0:
        active.append(f"state_noise(std={cfg.state_noise_std})")
    if cfg.action_noise_std > 0:
        active.append(f"action_noise(std={cfg.action_noise_std})")
    return active


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------

@_flatten_5d
def grayscale(imgs: torch.Tensor) -> torch.Tensor:
    """RGB to grayscale (replicated to 3 channels). ITU-R BT.601 weights."""
    weights = torch.tensor([0.299, 0.587, 0.114], device=imgs.device, dtype=imgs.dtype)
    gray = (imgs * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    return gray.expand_as(imgs)


@_flatten_5d
def camera_noise(
    imgs: torch.Tensor,
    noise_std: float,
    blur_sigma: tuple[float, float],
    blur_p: float,
    jpeg_quality: tuple[int, int],
    jpeg_p: float,
) -> torch.Tensor:
    """Simulate sensor noise, defocus blur, and JPEG compression."""
    if noise_std > 0:
        imgs = imgs + torch.randn_like(imgs) * noise_std
    if random.random() < blur_p:
        sigma = random.uniform(*blur_sigma)
        kernel_size = max(int(sigma * 4) | 1, 3)
        imgs = _gaussian_blur(imgs, kernel_size, sigma)
    if random.random() < jpeg_p:
        imgs = _jpeg_compress(imgs, random.randint(*jpeg_quality))
    return imgs.clamp(0.0, 1.0)


@_flatten_5d
def random_erase(
    imgs: torch.Tensor,
    p: float,
    scale: tuple[float, float],
) -> torch.Tensor:
    """Zero out random rectangular patches."""
    B, _C, H, W = imgs.shape
    for i in range(B):
        if random.random() > p:
            continue
        erase_area = random.uniform(*scale) * H * W
        aspect = random.uniform(0.3, 3.3)
        eh = int(round((erase_area * aspect) ** 0.5))
        ew = int(round((erase_area / aspect) ** 0.5))
        if eh >= H or ew >= W:
            continue
        top = random.randint(0, H - eh)
        left = random.randint(0, W - ew)
        imgs[i, :, top : top + eh, left : left + ew] = 0.0
    return imgs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_blur(imgs: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=imgs.device, dtype=imgs.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    pad = kernel_size // 2
    B, C, H, W = imgs.shape
    x = imgs.reshape(B * C, 1, H, W)
    x = F.conv2d(x, kernel_1d.view(1, 1, -1, 1), padding=(pad, 0))
    x = F.conv2d(x, kernel_1d.view(1, 1, 1, -1), padding=(0, pad))
    return x.reshape(B, C, H, W)


def _jpeg_compress(imgs: torch.Tensor, quality: int) -> torch.Tensor:
    import numpy as np
    from PIL import Image

    device = imgs.device
    result = []
    for img in imgs:
        arr = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        pil_img = Image.fromarray(arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        compressed = Image.open(buf)
        t = torch.from_numpy(np.array(compressed, dtype=np.float32) / 255.0)
        result.append(t.permute(2, 0, 1))
    return torch.stack(result).to(device)
