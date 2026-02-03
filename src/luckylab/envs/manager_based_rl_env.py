from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from luckyrobots import LuckyRobots, ObservationResponse

from ..configs.domain_randomization import PhysicsDRCfg
from ..entity import Entity, EntityCfg, Scene
from ..managers.action_manager import ActionManager, NullActionManager
from ..managers.command_manager import CommandManager, NullCommandManager
from ..managers.curriculum_manager import CurriculumManager, EpisodeMetrics, NullCurriculumManager
from ..managers.manager_term_config import (
    ActionTermCfg,
    CommandTermCfg,
    CurriculumTermCfg,
    ObservationGroupCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from ..managers.observation_manager import ObservationManager
from ..managers.reward_manager import RewardManager
from ..managers.termination_manager import TerminationManager

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ManagerBasedRlEnvCfg:
    """Configuration for a manager-based RL environment.

    Combines base environment settings with RL-specific configuration
    (rewards, terminations, commands, curriculum).

    This follows the mjlab pattern where all MDP components are defined
    as dicts mapping term names to their configs.
    """

    # Required fields (no defaults, must be provided).
    decimation: int
    """Number of simulation steps per environment step."""

    # Multi-environment settings.
    num_envs: int = 1
    """Number of parallel environments."""

    # Scene and robot (luckyrobots-specific).
    scene: str = "velocity"
    """Scene name to load."""
    task: str = "locomotion"
    """Task name."""
    robot: str = "unitreego1"
    """Robot name."""

    # Connection (luckyrobots-specific).
    host: str = "172.24.160.1"
    """gRPC host address."""
    port: int = 50051
    """gRPC port."""
    timeout_s: float = 120.0
    """Connection timeout in seconds."""
    skip_launch: bool = True
    """Whether to skip launching the engine (connect to existing)."""

    # Environment settings.
    seed: int | None = None
    """Random seed (None = random)."""
    debug: bool = False
    """Enable debug mode."""
    render_mode: str = "rgb_array"
    """Rendering mode."""

    # Episode settings.
    episode_length_s: float = 20.0
    """Maximum episode length in seconds."""
    sim_dt: float = 0.02
    """Simulation timestep in seconds (50Hz)."""
    is_finite_horizon: bool = False
    """Whether the task has a finite horizon (affects bootstrapping)."""

    # MDP configs - dicts of term configs (mjlab pattern).
    observations: dict[str, ObservationGroupCfg] = field(default_factory=dict)
    """Observation groups: name -> group config."""

    actions: dict[str, ActionTermCfg] = field(default_factory=dict)
    """Action terms: name -> config."""

    rewards: dict[str, RewardTermCfg] = field(default_factory=dict)
    """Reward terms: name -> config."""

    terminations: dict[str, TerminationTermCfg] = field(default_factory=dict)
    """Termination terms: name -> config."""

    commands: dict[str, CommandTermCfg] | None = None
    """Command terms: name -> config. None = no commands."""

    curriculum: dict[str, CurriculumTermCfg] | None = None
    """Curriculum terms: name -> config. None = no curriculum."""

    physics_dr: PhysicsDRCfg = field(default_factory=PhysicsDRCfg)
    """Physics domain randomization config."""

    # Derived properties.
    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in steps."""
        return int(self.episode_length_s / (self.sim_dt * self.decimation))

    @property
    def step_dt(self) -> float:
        """Environment step size (sim_dt * decimation)."""
        return self.sim_dt * self.decimation


class ManagerBasedRlEnv:
    """Manager-based RL environment for LuckyRobots.

    This environment connects to LuckyRobots via gRPC and provides a torch-native
    interface. All behavior (rewards, terminations, etc.) is driven by the provided
    configuration following the mjlab pattern.

    Rewards and terminations are defined as dicts mapping names to configs.
    Each config specifies a function reference and parameters.

    Returns torch tensors for all outputs.
    """

    def __init__(self, cfg: ManagerBasedRlEnvCfg, device: str = "cpu") -> None:
        """Initialize the environment.

        Args:
            cfg: Environment configuration.
            device: Torch device for computation.
        """
        self.cfg = cfg
        self.render_mode = cfg.render_mode
        self._num_envs = cfg.num_envs
        self._device = torch.device(device)

        # Initialize LuckyRobots connection
        self.luckyrobots = LuckyRobots(host=cfg.host, port=cfg.port)

        # Episode state (per-env tensors for multi-env support)
        self.episode_length_buf = torch.zeros(self._num_envs, dtype=torch.int64, device=self._device)
        self.common_step_counter = 0  # Total steps across all episodes (for curriculum)
        self.current_action: torch.Tensor | None = None
        self.last_action: torch.Tensor | None = None
        self.latest_observation: ObservationResponse | None = None

        # Schema info (populated after connection)
        self._observation_names: list[str] | None = None
        self._engine_observation_size: int = 0
        self._command_dim: int = 4  # Commands from LuckyLab CommandManager

        # Get robot config for joint information
        robot_config = LuckyRobots.get_robot_config(cfg.robot)
        action_limits = robot_config["action_space"]["actuator_limits"]
        self.num_joints = len(action_limits)
        self.action_low = torch.tensor(
            [limit["lower"] for limit in action_limits], dtype=torch.float32, device=self._device
        )
        self.action_high = torch.tensor(
            [limit["upper"] for limit in action_limits], dtype=torch.float32, device=self._device
        )

        # Connect to LuckyRobots and fetch schema FIRST to get actual observation size
        self._start_or_connect()
        self._fetch_observation_schema()

        # Initialize scene with robot entity (mjlab pattern: env.scene["robot"].data.*)
        joint_names = self._get_joint_names()
        actuator_names = self._get_actuator_names()
        self.scene = Scene()
        robot_entity = Entity(
            cfg=EntityCfg(),
            num_envs=self._num_envs,
            num_joints=self.num_joints,
            joint_names=joint_names,
            device=self._device,
            actuator_names=actuator_names,
        )
        self.scene.add("robot", robot_entity)

        # Set default joint positions from robot config
        # This is used by JointPositionAction when use_default_offset=True
        default_positions = [
            limit.get("default", 0.0) for limit in robot_config["action_space"]["actuator_limits"]
        ]
        robot_entity.data.set_default_joint_pos(default_positions)
        logger.info(f"Default joint positions: {default_positions}")

        # Validate which observations are available from the engine
        # This logs warnings for missing privileged observations (foot sensors, etc.)
        if self._observation_names:
            robot_entity.data.validate_observations(self._observation_names)

        # Calculate fallback policy observation size (used if no observation manager):
        # [commands (from CommandManager)] + [engine_state] + [last_act (from ActionManager)]
        policy_obs_size = self._command_dim + self._engine_observation_size + self.num_joints
        logger.info(
            f"Fallback observation size: {policy_obs_size} "
            f"(commands={self._command_dim} + engine={self._engine_observation_size} + last_act={self.num_joints})"
        )

        # Setup spaces with fallback observation size (may be updated by observation manager)
        self._setup_spaces(robot_config, policy_obs_size)

        # Initialize action manager for action processing (raw → joint positions)
        # Uses mjlab pattern: actions dict with ActionTermCfg containing class_type
        if cfg.actions:
            self.action_manager = ActionManager(cfg.actions, self)
            logger.info(f"Action manager: {self.action_manager.active_terms} ({self.action_manager.total_action_dim} dims)")
        else:
            self.action_manager = NullActionManager(self.num_joints, self._num_envs, self._device)
            logger.info("Action manager: disabled (no actions config)")

        # Initialize command manager using mjlab pattern (class_type instantiation)
        # Must be done after connection so we have num_envs properly set up
        if cfg.commands:
            self.command_manager = CommandManager(cfg.commands, self)
            logger.info(f"Command manager: {self.command_manager.active_terms}")
        else:
            self.command_manager = NullCommandManager()
            logger.info("Command manager: disabled (no commands config)")

        # Initialize reward manager (if reward terms are defined)
        self.reward_manager: RewardManager | None = None
        if cfg.rewards:
            self.reward_manager = RewardManager(cfg.rewards, self)
            logger.info(f"Reward manager: {len(self.reward_manager.active_terms)} terms")

        # Initialize termination manager (if termination terms are defined)
        self.termination_manager: TerminationManager | None = None
        if cfg.terminations:
            self.termination_manager = TerminationManager(cfg.terminations, self)
            logger.info(f"Termination manager: {len(self.termination_manager.active_terms)} terms")

        # Initialize curriculum manager (if curriculum terms are defined)
        self.curriculum_manager: CurriculumManager | NullCurriculumManager
        if cfg.curriculum:
            self.curriculum_manager = CurriculumManager(cfg.curriculum, self)
            logger.info(f"Curriculum manager: {len(self.curriculum_manager.active_terms)} terms")
        else:
            self.curriculum_manager = NullCurriculumManager()
            logger.info("Curriculum manager: disabled (no curriculum config)")

        # Initialize observation manager (required)
        if not cfg.observations:
            raise ValueError(
                "observations config is required. Define observation groups in your environment config."
            )
        self.observation_manager = ObservationManager(cfg.observations, self)
        logger.info(f"Observation manager: {self.observation_manager.active_groups} groups")

        # Update observation space with actual computed dimension
        try:
            policy_obs_dim = self.observation_manager.get_observation_dim("policy")
            self._obs_shape = (policy_obs_dim,)
            if self.observation_space is not None:
                import numpy as np
                from gymnasium import spaces

                self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(policy_obs_dim,),
                    dtype=np.float32,
                )
            logger.info(f"Updated observation space: dim={policy_obs_dim}")
        except Exception as e:
            logger.warning(f"Could not compute observation dimension: {e}")

        # Episode metrics tracking
        self.episode_metrics = EpisodeMetrics()
        self.episode_count = 0

        # Task name for prefixing metrics (capitalize first letter)
        self.task_name = cfg.task.replace("_", " ").title().replace(" ", "")

        # Extras dict for logging (mjlab pattern)
        self.extras: dict = {"log": {}}

        # Log configured rewards and terminations
        if cfg.rewards:
            logger.info(f"Configured rewards: {list(cfg.rewards.keys())}")
        if cfg.terminations:
            logger.info(f"Configured terminations: {list(cfg.terminations.keys())}")

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self._num_envs

    @property
    def device(self) -> torch.device:
        """Device for computation."""
        return self._device

    @property
    def observation_space_shape(self) -> tuple[int, ...]:
        """Shape of the observation space."""
        return self._obs_shape

    @property
    def action_space_shape(self) -> tuple[int, ...]:
        """Shape of the action space."""
        return self._action_shape

    def _setup_spaces(self, robot_config: dict, obs_dim: int) -> None:
        """Set up action and observation space shapes.

        Args:
            robot_config: Robot configuration dict.
            obs_dim: Observation dimension (from schema or fallback).
        """
        action_limits = robot_config["action_space"]["actuator_limits"]
        action_dim = len(action_limits)

        self._action_shape = (action_dim,)
        self._obs_shape = (obs_dim,)

        # Keep gymnasium-compatible spaces for wrapper compatibility
        try:
            import gymnasium as gym
            import numpy as np
            from gymnasium import spaces

            # When actions are configured, policy outputs normalized [-1, 1] actions
            # The ActionManager converts these to actual joint positions
            if self.cfg.actions:
                self.action_space = spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(action_dim,),
                    dtype=np.float32,
                )
            else:
                # Fallback to raw joint position limits
                self.action_space = spaces.Box(
                    low=np.array([limit["lower"] for limit in action_limits], dtype=np.float32),
                    high=np.array([limit["upper"] for limit in action_limits], dtype=np.float32),
                    shape=(action_dim,),
                    dtype=np.float32,
                )

            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )
        except ImportError:
            # gymnasium not installed, spaces won't be available
            self.action_space = None
            self.observation_space = None

    def _get_joint_names(self) -> list[str]:
        """Get joint names from robot config."""
        robot_config = LuckyRobots.get_robot_config(self.cfg.robot)
        action_limits = robot_config["action_space"]["actuator_limits"]
        return [limit.get("name", f"joint_{i}") for i, limit in enumerate(action_limits)]

    def _get_actuator_names(self) -> list[str]:
        """Get actuator names from robot config.

        Returns actuator names from action_space.actuator_names if available,
        otherwise falls back to names from action_space.actuator_limits.
        """
        robot_config = LuckyRobots.get_robot_config(self.cfg.robot)
        action_space = robot_config["action_space"]

        # Prefer explicit actuator_names list if available
        if "actuator_names" in action_space:
            return list(action_space["actuator_names"])

        # Fall back to names from actuator_limits
        action_limits = action_space["actuator_limits"]
        return [limit.get("name", f"actuator_{i}") for i, limit in enumerate(action_limits)]

    def _start_or_connect(self) -> None:
        """Launch LuckyEngine or connect to existing instance."""
        if self.cfg.skip_launch:
            logger.info("Connecting to existing LuckyEngine instance...")
            self.luckyrobots.connect(timeout_s=self.cfg.timeout_s, robot=self.cfg.robot)
        else:
            logger.info("Launching LuckyEngine...")
            self.luckyrobots.start(
                scene=self.cfg.scene,
                robot=self.cfg.robot,
                task=self.cfg.task,
                timeout_s=self.cfg.timeout_s,
            )

        # Set simulation mode to "fast" for RL training (runs physics as fast as possible)
        self.luckyrobots.set_simulation_mode("fast")
        logger.info("Set simulation mode to 'fast' for RL training")

    def _fetch_observation_schema(self) -> None:
        """Fetch observation schema from engine (required).

        The schema contains observation_names and observation_size from the engine.
        Engine observations no longer include commands or last_act - those are
        managed by LuckyLab's CommandManager and ActionManager respectively.

        Raises:
            RuntimeError: If schema cannot be fetched from engine.
        """
        if self.luckyrobots.engine_client is None:
            raise RuntimeError(
                "Cannot fetch observation schema: engine client not connected. "
                "Ensure LuckyEngine is running and connection is established."
            )

        try:
            schema_resp = self.luckyrobots.engine_client.get_agent_schema()
            if not hasattr(schema_resp, "schema") or not schema_resp.schema:
                raise RuntimeError("Engine returned empty schema response.")

            self._observation_names = list(schema_resp.schema.observation_names)
            self._engine_observation_size = schema_resp.schema.observation_size

            logger.info(
                f"Fetched observation schema: {len(self._observation_names)} names, "
                f"engine_observation_size={self._engine_observation_size}"
            )
            logger.info(f"Observation names: {self._observation_names}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch observation schema from engine: {e}. "
                "Ensure LuckyEngine is running with a valid agent."
            ) from e

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[torch.Tensor, dict]:
        """Reset the environment.

        On reset, the extras["log"] dict is populated with episode statistics
        from all managers. This follows the mjlab pattern for logging.

        Returns:
            Tuple of (observation tensor, info dict).
        """
        if seed is not None:
            torch.manual_seed(seed)

        # For single-env, use env_idx=0. Multi-env would reset specific envs.
        env_ids = torch.tensor([0], dtype=torch.long, device=self._device)

        # Initialize log dict for this reset (mjlab pattern)
        self.extras["log"] = {}

        # Collect episode stats from managers before resetting
        if self.reward_manager is not None:
            info = self.reward_manager.reset(env_ids)
            self.extras["log"].update(info)

        if self.termination_manager is not None:
            info = self.termination_manager.reset(env_ids)
            self.extras["log"].update(info)

        # Add episode-level metrics before resetting
        if self.episode_metrics.steps > 0:
            episode_data = self.episode_metrics.finalize()
            self.extras["log"]["Episode/total_reward"] = episode_data["total_reward"]
            self.extras["log"]["Episode/length"] = float(episode_data["episode_length"])
            self.extras["log"]["Episode/terminated"] = float(episode_data["terminated"])
            self.extras["log"]["Episode/truncated"] = float(episode_data["truncated"])
            self.extras["log"]["Episode/survived"] = float(episode_data["survived"])

        # Increment episode count
        self.episode_count += 1

        # Update curriculum based on step count (mjlab pattern)
        self.curriculum_manager.compute(env_ids)

        # Reset episode state
        self.episode_length_buf[env_ids] = 0
        self.current_action = None
        self.last_action = None
        self.episode_metrics.reset()
        self.action_manager.reset(env_ids)
        self.command_manager.reset(env_ids)

        # Reset environment via gRPC with domain randomization
        observation = self.luckyrobots.reset(randomization_cfg=self.cfg.physics_dr)
        self.latest_observation = observation

        # Update entity data from observation for observation functions
        if observation.observation:
            obs_tensor = torch.tensor(observation.observation, dtype=torch.float32, device=self._device)
            self.scene["robot"].data.update_from_observation(obs_tensor, self._observation_names)

        # Reset observation manager buffers (delay/history)
        self.observation_manager.reset(env_ids)

        # Compute observation using observation manager
        obs = self.observation_manager.compute("policy")
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # Ensure (num_envs, obs_dim) shape

        # Build info dict with command and log reference
        cmd = self.command_manager.get_command("twist")
        cmd_list = (
            cmd[0].tolist()
            if cmd is not None and cmd.dim() == 2
            else (cmd.tolist() if cmd is not None else [])
        )
        info = {
            "command": cmd_list,
            "log": self.extras["log"],
        }

        return obs, info

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Execute one step in the environment.

        Args:
            action: Action tensor.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
            All tensors have shape (num_envs,) or (num_envs, dim).
        """
        env_idx = 0  # Single-env for now

        # Ensure action has batch dimension (num_envs, action_dim)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        self.episode_length_buf[env_idx] += 1
        self.common_step_counter += 1  # Track total steps for curriculum

        if self.current_action is not None:
            self.last_action = self.current_action.clone()
        else:
            self.last_action = torch.zeros_like(action)
        self.current_action = action.clone()

        # Process action through actuator layer (raw [-1,1] → joint positions)
        joint_positions = self.action_manager.process_action(action)

        # Convert to list for gRPC (send first env's action for single-engine case)
        joint_list = joint_positions[env_idx].detach().cpu().tolist()

        # Synchronous RL step: send action, wait for physics, get observation in one RPC
        observation = self.luckyrobots.step(actions=joint_list)
        self.latest_observation = observation

        # Update entity data from observation (mjlab pattern: env.scene["robot"].data.*)
        if observation.observation:
            obs_tensor = torch.tensor(observation.observation, dtype=torch.float32, device=self._device)
            self.scene["robot"].data.update_from_observation(obs_tensor, self._observation_names)

        # Update commands
        self.command_manager.compute(self.cfg.sim_dt)

        # Compute reward using reward manager (accesses env.data directly)
        cmd = self.command_manager.get_command("twist")
        cmd_list = (
            cmd[env_idx].tolist()
            if cmd is not None and cmd.dim() == 2
            else (cmd.tolist() if cmd is not None else [])
        )
        info: dict[str, Any] = {"command": cmd_list}

        if self.reward_manager is not None:
            reward = self.reward_manager.compute(dt=self.cfg.step_dt)
            if reward.dim() == 0:
                reward = reward.unsqueeze(0)  # Ensure (num_envs,) shape
            info["reward_details"] = {
                name: value for name, [value] in self.reward_manager.get_active_iterable_terms(env_idx)
            }
        else:
            reward = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)

        # Compute termination using termination manager (accesses env.data directly)
        if self.termination_manager is not None:
            self.termination_manager.compute()
            terminated = self.termination_manager.terminated  # Shape: (num_envs,)
            info["termination_reasons"] = self.termination_manager.termination_reasons(env_idx)
            info["termination_reason"] = (
                info["termination_reasons"][0] if info["termination_reasons"] else "none"
            )
            info["is_success"] = False
        else:
            terminated = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)

        # Compute observation using observation manager
        # Noise, delay, and history are applied automatically
        obs = self.observation_manager.compute("policy")
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # Ensure (num_envs, obs_dim) shape

        # Check truncation (timeout) - shape: (num_envs,)
        truncated = self.episode_length_buf >= self.cfg.max_episode_length

        # Update episode metrics (using env_idx=0 for single env metrics tracking)
        self.episode_metrics.add_step()
        reward_scalar = reward[env_idx].item() if isinstance(reward, torch.Tensor) else reward
        self.episode_metrics.add_reward(reward_scalar)
        term_val = terminated[env_idx].item() if isinstance(terminated, torch.Tensor) else terminated
        trunc_val = truncated[env_idx].item() if isinstance(truncated, torch.Tensor) else truncated
        if term_val:
            self.episode_metrics.set_terminated(info.get("termination_reason", ""))
        if trunc_val:
            self.episode_metrics.set_truncated()

        # Ensure all outputs are tensors with correct shapes
        # obs: (num_envs, obs_dim), reward: (num_envs,), terminated: (num_envs,), truncated: (num_envs,)
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor([reward] * self._num_envs, dtype=torch.float32, device=self._device)
        if not isinstance(terminated, torch.Tensor):
            terminated = torch.tensor([terminated] * self._num_envs, dtype=torch.bool, device=self._device)
        if not isinstance(truncated, torch.Tensor):
            truncated = torch.tensor([truncated] * self._num_envs, dtype=torch.bool, device=self._device)

        return obs, reward, terminated, truncated, info

    def render(self) -> torch.Tensor | None:
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return torch.zeros((480, 640, 3), dtype=torch.uint8, device=self._device)
        return None

    def close(self) -> None:
        """Close the environment."""
        if self.luckyrobots:
            self.luckyrobots.close(stop_engine=not self.cfg.skip_launch)
        logger.info("Environment closed")

    def get_episode_log(self) -> dict:
        """Get the current episode's log data."""
        return self.extras.get("log", {})

    def get_training_metrics(self) -> dict:
        """Get current training metrics for logging."""
        prefix = self.task_name
        metrics = {
            f"{prefix}/total_steps": float(self.common_step_counter),
            f"{prefix}/total_episodes": float(self.episode_count),
            f"{prefix}/current_episode_length": float(self.episode_length_buf[0].item()),
        }

        if self.reward_manager is not None:
            for name, value in self.reward_manager.episode_sums(0).items():
                metrics[f"{prefix}_Reward/{name}"] = value

        return metrics

    def log_metric(self, name: str, value: float) -> None:
        """Add a custom metric to the current episode's log."""
        key = f"{self.task_name}/{name}"
        self.extras["log"][key] = value

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Add multiple custom metrics to the current episode's log."""
        for name, value in metrics.items():
            self.log_metric(name, value)
