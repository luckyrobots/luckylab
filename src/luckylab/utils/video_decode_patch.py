"""Patch lerobot video decoding to handle short videos gracefully.

When a video has fewer frames than its parquet expects, lerobot raises
FrameTimestampError or IndexError. This module replaces the decode functions
with versions that clamp frame indices and use the nearest available frame
instead of raising.

Call ``install()`` once before any LeRobotDataset access.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torchvision

_installed = False


@dataclass
class Stats:
    tolerance_violations: int = 0
    index_errors: int = 0
    pyav_fallbacks: int = 0
    max_deviation_s: float = 0.0
    affected_videos: Counter = field(default_factory=Counter)

    def record(self, video_path: str, *, kind: str, deviation_s: float = 0.0) -> None:
        if kind == "tolerance":
            self.tolerance_violations += 1
            self.max_deviation_s = max(self.max_deviation_s, deviation_s)
        elif kind == "index":
            self.index_errors += 1
        elif kind == "fallback":
            self.pyav_fallbacks += 1
        self.affected_videos[video_path] += 1

    def summary(self) -> str:
        if not self.tolerance_violations and not self.index_errors:
            return "[video_decode_patch] No issues encountered."
        top = self.affected_videos.most_common(5)
        offenders = "\n".join(f"    {c:4d}x  {Path(p).name}" for p, c in top)
        return (
            f"[video_decode_patch] Summary:\n"
            f"  Tolerance violations: {self.tolerance_violations}\n"
            f"  Index errors:         {self.index_errors}\n"
            f"  PyAV fallbacks:       {self.pyav_fallbacks}\n"
            f"  Max deviation:        {self.max_deviation_s:.3f}s\n"
            f"  Affected videos:      {len(self.affected_videos)}\n"
            f"  Top offenders:\n{offenders}"
        )


_stats = Stats()


def get_stats() -> Stats:
    return _stats


def _check_tolerance(query_ts, loaded_ts, tolerance_s, video_path):
    """Return argmin indices; record stats if tolerance exceeded."""
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)
    if not (min_ < tolerance_s).all():
        _stats.record(str(video_path), kind="tolerance", deviation_s=min_.max().item())
    return argmin_


def _clamp(indices: list[int], num_frames: int | None) -> list[int]:
    if num_frames is None:
        return [max(0, i) for i in indices]
    last = num_frames - 1
    return [min(max(0, i), last) for i in indices]


def _get_num_frames(metadata) -> int | None:
    for attr in ("num_frames", "n_frames", "num_video_frames",
                 "frames", "video_num_frames", "total_frames"):
        val = getattr(metadata, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    return None


def _decode_torchvision(video_path, timestamps, tolerance_s, backend="pyav", log_loaded_timestamps=False):
    video_path = str(video_path)
    torchvision.set_video_backend(backend)
    reader = torchvision.io.VideoReader(video_path, "video")
    reader.seek(min(timestamps), keyframes_only=(backend == "pyav"))

    loaded_frames, loaded_ts = [], []
    for frame in reader:
        loaded_frames.append(frame["data"])
        loaded_ts.append(frame["pts"])
        if frame["pts"] >= max(timestamps):
            break
    if backend == "pyav":
        reader.container.close()

    argmin_ = _check_tolerance(torch.tensor(timestamps), torch.tensor(loaded_ts), tolerance_s, video_path)
    return torch.stack([loaded_frames[i] for i in argmin_]).float() / 255


def _decode_torchcodec(video_path, timestamps, tolerance_s, log_loaded_timestamps=False, decoder_cache=None):
    import lerobot.datasets.video_utils as vu

    decoder = (decoder_cache or vu._default_decoder_cache).get_decoder(str(video_path))
    meta = decoder.metadata
    fps = meta.average_fps
    raw = [round(ts * fps) for ts in timestamps]
    indices = _clamp(raw, _get_num_frames(meta))

    try:
        batch = decoder.get_frames_at(indices=indices)
    except IndexError:
        _stats.record(str(video_path), kind="index")
        try:
            dur = vu.get_video_duration_in_s(video_path)
            indices = _clamp(raw, max(1, int(math.ceil(dur * float(fps)))))
            batch = decoder.get_frames_at(indices=indices)
        except Exception:
            _stats.record(str(video_path), kind="fallback")
            return _decode_torchvision(video_path, timestamps, tolerance_s, backend="pyav")

    loaded_ts = torch.tensor([p.item() for p in batch.pts_seconds])
    argmin_ = _check_tolerance(torch.tensor(timestamps), loaded_ts, tolerance_s, video_path)
    return torch.stack([batch.data[i] for i in argmin_]).float() / 255


def _decode(video_path, timestamps, tolerance_s, backend=None):
    import lerobot.datasets.video_utils as vu
    backend = backend or vu.get_safe_default_codec()
    if backend == "torchcodec":
        return _decode_torchcodec(video_path, timestamps, tolerance_s)
    if backend in ("pyav", "video_reader"):
        return _decode_torchvision(video_path, timestamps, tolerance_s, backend)
    raise ValueError(f"Unsupported video backend: {backend}")


def install() -> None:
    """Replace lerobot's video decode functions with tolerant versions (idempotent)."""
    global _installed
    if _installed:
        return
    import lerobot.datasets.video_utils as vu
    vu.decode_video_frames = _decode
    vu.decode_video_frames_torchvision = _decode_torchvision
    vu.decode_video_frames_torchcodec = _decode_torchcodec
    _installed = True
