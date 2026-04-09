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
    episodes: list[int] | None = None
    """List of episode indices to use. None = use all episodes.
    Example: list(range(2, 152)) to skip the first 2 episodes."""

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

    # Data augmentation
    grayscale: bool = False
    """Convert images to grayscale (replicated to 3 channels).
    Removes color domain gap entirely, forcing the policy to rely on geometry."""

    camera_noise: bool = False
    """Apply simulated camera noise: Gaussian sensor noise, slight blur, and
    JPEG compression artifacts."""
    camera_noise_std: float = 0.02
    """Gaussian sensor noise std (in [0,1] pixel range). 0.02 = normal indoor webcam."""
    camera_blur_sigma: tuple[float, float] = (0.1, 1.0)
    """Gaussian blur sigma range, sampled uniformly. Simulates slight defocus."""
    camera_blur_p: float = 0.3
    """Probability of applying blur to each image."""
    camera_jpeg_quality: tuple[int, int] = (70, 95)
    """JPEG compression quality range, sampled uniformly. Lower = more artifacts."""
    camera_jpeg_p: float = 0.5
    """Probability of applying JPEG compression to each image."""

    random_erasing: bool = False
    """Randomly zero out rectangular patches of images. Forces distributed
    visual features, preventing reliance on sim-specific visual details."""
    random_erasing_p: float = 0.3
    """Probability of erasing a patch per image."""
    random_erasing_scale: tuple[float, float] = (0.02, 0.15)
    """Range of proportion of image area to erase."""

    state_noise_std: float = 0.0
    """Gaussian noise std added to observation.state (radians).
    Simulates real servo encoder noise and backlash.
    0.035 = realistic for Feetech STS3215 servos (~2 degrees)."""

    action_noise_std: float = 0.0
    """Gaussian noise std added to action labels (radians).
    Simulates imperfect action execution (commanded vs actual position).
    0.05 = realistic for STS3215 under load (~3 degrees)."""

    # Eval connection info (for deploying back to LuckyEngine)
    scene: str = ""
    """LuckyEngine scene to use for evaluation."""
    robot: str = ""
    """Robot type for evaluation."""
    task: str = ""
    """Task name (e.g. 'pickandplace')."""
    host: str = "localhost"
    """LuckyEngine gRPC host."""
    port: int = 50051
    """LuckyEngine gRPC port."""
    timeout_s: float = 120.0
    """Connection timeout in seconds."""
    step_timeout_s: float = 30.0
    """Per-step physics timeout in seconds."""
    skip_launch: bool = True
    """Skip launching engine (connect to existing)."""
