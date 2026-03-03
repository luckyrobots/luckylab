"""IL training dispatcher — routes to the LeRobot backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from luckylab.il.config import IlRunnerCfg


def train(il_cfg: IlRunnerCfg, device: str = "cpu") -> None:
    """Train an IL policy using the LeRobot backend.

    Args:
        il_cfg: IL runner configuration (includes policy type, dataset info, etc.).
        device: Torch device string.
    """
    from luckylab.il.lerobot.trainer import train as lerobot_train

    lerobot_train(il_cfg, device)


def load_policy(checkpoint_path: str, il_cfg: IlRunnerCfg, device: str = "cpu"):
    """Load a trained IL policy with its preprocessor and postprocessor.

    Args:
        checkpoint_path: Path to saved policy checkpoint directory.
        il_cfg: IL runner configuration.
        device: Torch device string.

    Returns:
        Tuple of (policy, preprocessor, postprocessor).
    """
    from luckylab.il.lerobot.trainer import load_policy as lerobot_load

    return lerobot_load(checkpoint_path, il_cfg, device)
