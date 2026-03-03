#!/usr/bin/env python3
"""Visualize a LeRobot dataset in Rerun.

Loads a LeRobotDataset, iterates through an episode, and logs every frame
to rerun with proper timeline (frame_index + timestamp).

Usage:
    python -m luckylab.scripts.visualize_dataset \
        --repo-id lerobot/aloha_static_cups_open --episode-index 0

    # Save to .rrd file instead of spawning viewer
    python -m luckylab.scripts.visualize_dataset \
        --repo-id lerobot/aloha_static_cups_open --episode-index 0 --save recording.rrd
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from luckylab.utils.logging import print_header, print_info

LUCKYROBOTS_DATA_HOME = Path(os.getenv("LUCKYROBOTS_DATA_HOME", Path.home() / ".luckyrobots" / "data"))


@dataclass(frozen=True)
class VisualizeDatasetConfig:
    """Dataset visualization configuration."""

    repo_id: str
    """HuggingFace repo ID or local dataset name (e.g. piper/pickandplace)."""
    root: str | None = None
    """Local root directory for the dataset. Auto-detected from LUCKYROBOTS_DATA_HOME if not set."""
    episode_index: int = 0
    """Which episode to visualize."""
    save: str | None = None
    """Save rerun recording to .rrd file instead of spawning viewer."""
    web: bool = False
    """Serve a web viewer instead of spawning native viewer."""


def main() -> int:
    cfg = tyro.cli(VisualizeDatasetConfig)

    print_header("Dataset Visualization")
    print_info(f"Loading dataset: {cfg.repo_id}, episode {cfg.episode_index}")

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print_info(
            "lerobot is required for dataset visualization. "
            "Install with: uv sync --group il",
            color="red",
        )
        return 1

    from luckylab.utils import video_decode_patch
    from luckylab.utils.rerun_logger import RerunLogger

    video_decode_patch.install()

    # Resolve dataset root (explicit > LUCKYROBOTS_DATA_HOME > hub)
    ds_kwargs: dict = {"episodes": [cfg.episode_index]}
    root = cfg.root
    if root is None:
        candidate = LUCKYROBOTS_DATA_HOME / cfg.repo_id
        if candidate.exists():
            root = str(candidate)
    if root:
        ds_kwargs["root"] = root
        print_info(f"Using local dataset root: {root}")
    dataset = LeRobotDataset(cfg.repo_id, **ds_kwargs)
    num_frames = len(dataset)

    print_info(f"Episode {cfg.episode_index}: {num_frames} frames")

    # Detect camera keys from the first sample
    camera_keys = []
    sample0 = dataset[0]
    for key in sample0:
        if "image" in key.lower():
            camera_keys.append(key)
    if camera_keys:
        print_info(f"Camera keys: {camera_keys}")

    with RerunLogger(
        app_id=f"luckylab/dataset/{cfg.repo_id.replace('/', '_')}",
        save_path=cfg.save,
        web=cfg.web,
    ) as rr_log:
        rr = rr_log._rr

        for i in range(num_frames):
            sample = dataset[i]

            # Set time from frame index and timestamp
            frame_idx = sample["frame_index"].item() if hasattr(sample["frame_index"], "item") else int(sample["frame_index"])
            rr.set_time("frame_index", sequence=frame_idx)
            if "timestamp" in sample:
                ts = sample["timestamp"].item() if hasattr(sample["timestamp"], "item") else float(sample["timestamp"])
                rr.set_time("timestamp", timestamp=ts)

            # Log camera images
            for cam_key in camera_keys:
                if cam_key in sample:
                    img = sample[cam_key]
                    if hasattr(img, "numpy"):
                        img = img.numpy()
                    img = np.asarray(img)
                    # Normalize float [0,1] images to uint8
                    if img.dtype in (np.float32, np.float64) and img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    rr_log.log_image(cam_key, img)

            # Log state observations
            state_key = "observation.state"
            if state_key in sample:
                state = sample[state_key]
                if hasattr(state, "numpy"):
                    state = state.numpy()
                state = np.asarray(state).flatten()
                for j, val in enumerate(state):
                    rr_log.log_scalar(f"state/{j}", float(val))

            # Log actions
            if "action" in sample:
                action = sample["action"]
                if hasattr(action, "numpy"):
                    action = action.numpy()
                action = np.asarray(action).flatten()
                for j, val in enumerate(action):
                    rr_log.log_scalar(f"action/{j}", float(val))

    print_info("Visualization complete!")
    if cfg.web:
        url = rr_log._web_url or "http://localhost:9090"
        print_info(f"Web viewer: {url}")
        print_info("Press Ctrl+C to stop")
        try:
            import time

            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
