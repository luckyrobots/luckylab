"""IL (Imitation Learning) training configuration."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IlRunnerCfg:
    """
    Top-level IL training configuration.

    Policy architecture is selected by name (e.g., "act", "diffusion", "pi0", "smolvla").
    LeRobot's factory creates the policy config with appropriate defaults.
    Use ``policy_overrides`` to customize specific fields on the lerobot config.
    """

    # Policy selection
    policy: str = "act"
    """IL policy name — any policy registered in lerobot's factory
    (e.g., "act", "diffusion", "pi0", "pi0_fast", "smolvla", "xvla")."""

    policy_overrides: dict[str, Any] = field(default_factory=dict)
    """Key-value overrides applied to the lerobot policy config after factory creation.
    Example: {"chunk_size": 50, "dim_model": 256, "use_vae": False}."""

    # Dataset
    dataset_repo_id: str = ""
    """HuggingFace repo ID or local path to the LeRobotDataset."""
    dataset_root: str | None = None
    """Local root directory where LuckyEngine wrote data. Overrides default cache."""
    delta_timestamps: dict[str, list[float]] | None = None
    """Temporal observation stacking config. Maps key -> list of relative timestamps."""

    # Training
    batch_size: int = 8
    """Training batch size."""
    num_train_steps: int = 100_000
    """Total number of training gradient steps."""
    seed: int = 42
    """Random seed."""

    # Optimizer — None means "use the policy's recommended preset"
    learning_rate: float | None = None
    """Learning rate. None = use policy preset (e.g. 1e-5 for ACT, 1e-4 for Diffusion)."""
    weight_decay: float | None = None
    """Weight decay. None = use policy preset."""
    grad_clip_norm: float | None = None
    """Maximum gradient norm for clipping. None = use policy preset (typically 10.0)."""

    # Checkpoint
    save_freq: int = 10_000
    """Save checkpoint every N steps."""

    # Experiment
    experiment_name: str = "luckylab_il"
    """Experiment name."""
    directory: str = "runs"
    """Output directory."""

    # Logging
    wandb: bool = True
    """Enable wandb logging."""
    wandb_project: str = "luckylab"
    """W&B project name."""
    wandb_entity: str | None = None
    """W&B entity (team/user name)."""

    # Eval connection info (for deploying back to LuckyEngine)
    scene: str = ""
    """LuckyEngine scene to use for evaluation."""
    robot: str = ""
    """Robot type for evaluation."""
    host: str = "localhost"
    """LuckyEngine host for eval connection."""
    port: int = 8080
    """LuckyEngine port for eval connection."""
    simulation_mode: str = "realtime"
    """Simulation mode for evaluation (realtime or headless)."""
