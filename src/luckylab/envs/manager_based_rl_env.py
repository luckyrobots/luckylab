"""Manager-based RL environment.

This module defines both the configuration and implementation for manager-based
RL environments, following the mjlab pattern where all MDP components are
config-driven with direct function references.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from luckyrobots import LuckyRobots, ObservationResponse

from ..configs.domain_randomization import PhysicsDRCfg
from ..managers.command_manager import CommandManager, VelocityCommand
from ..managers.curriculum_manager import CurriculumTermManager, EpisodeMetrics
from ..managers.manager_term_config import (
    ActionTermCfg,
    CommandTermCfg,
    CurriculumTermCfg,
    ObservationGroupCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from ..managers.observation_manager import ObservationProcessor, ObservationProcessorCfg
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

    # Note: EventTermCfg is not used here because domain randomization
    # (physics params like mass, friction) must be done engine-side via gRPC.
    # When the engine exposes DR endpoints, we can add event support.

    # Domain randomization (luckyrobots-specific).
    physics_dr: PhysicsDRCfg = field(default_factory=PhysicsDRCfg)
    """Physics domain randomization config."""
    observation_dr: ObservationProcessorCfg = field(default_factory=ObservationProcessorCfg)
    """Observation domain randomization/processing config."""

    # Derived properties.
    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in steps."""
        return int(self.episode_length_s / (self.sim_dt * self.decimation))

    @property
    def step_dt(self) -> float:
        """Environment step size (sim_dt * decimation)."""
        return self.sim_dt * self.decimation


class ManagerBasedRlEnv(gym.Env):
    """Manager-based RL environment for LuckyRobots.

    This environment connects to LuckyRobots via gRPC and provides a standard
    Gymnasium interface. All behavior (rewards, terminations, etc.) is driven
    by the provided configuration following the mjlab pattern.

    Rewards and terminations are defined as dicts mapping names to configs.
    Each config specifies a function reference and parameters.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, cfg: ManagerBasedRlEnvCfg) -> None:
        """Initialize the environment.

        Args:
            cfg: Environment configuration.
        """
        super().__init__()
        self.cfg = cfg
        self.render_mode = cfg.render_mode

        # Initialize LuckyRobots connection
        self.luckyrobots = LuckyRobots(host=cfg.host, port=cfg.port)

        # Episode state
        self.step_count = 0
        self.common_step_counter = 0  # Total steps across all episodes (for curriculum)
        self.current_action: np.ndarray | None = None
        self.last_action: np.ndarray | None = None
        self.latest_observation: ObservationResponse | None = None

        # Get robot config for joint information
        robot_config = LuckyRobots.get_robot_config(cfg.robot)
        action_limits = robot_config["action_space"]["actuator_limits"]
        self.num_joints = len(action_limits)
        self.action_low = np.array([limit["lower"] for limit in action_limits], dtype=np.float32)
        self.action_high = np.array([limit["upper"] for limit in action_limits], dtype=np.float32)

        # Initialize observation parser (import here to avoid circular import)
        from ..tasks.velocity.mdp import ObservationParser

        self.obs_parser = ObservationParser(self.num_joints)

        # Initialize command manager
        self.command_manager = CommandManager()
        self.command_manager.add_command("velocity", VelocityCommand(cfg.commands))

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

        # Note: EventManager is not initialized here because domain randomization
        # requires engine-side support via gRPC. When DR endpoints are available,
        # event-based randomization can be added.

        # Initialize curriculum manager (if curriculum terms are defined)
        self.curriculum_manager: CurriculumTermManager | None = None
        if cfg.curriculum:
            self.curriculum_manager = CurriculumTermManager(cfg.curriculum)
            logger.info("Curriculum learning enabled")

        # Episode metrics tracking
        self.episode_metrics = EpisodeMetrics()
        self.episode_count = 0

        # Task name for prefixing metrics (capitalize first letter)
        self.task_name = cfg.task.replace("_", " ").title().replace(" ", "")

        # Extras dict for logging (mjlab pattern)
        # This is populated on reset with episode statistics
        self.extras: dict = {"log": {}}

        # Initialize observation processor for observation DR
        obs_config = robot_config["observation_space"]["actuator_limits"]
        obs_dim = len(obs_config)
        self.observation_processor = ObservationProcessor(obs_dim, cfg.observation_dr)
        self._obs_processor_initialized = False

        # Setup spaces
        self._setup_spaces(robot_config)

        # Connect to LuckyRobots
        self._start_or_connect()

        # Log configured rewards and terminations
        if cfg.rewards:
            logger.info(f"Configured rewards: {list(cfg.rewards.keys())}")
        if cfg.terminations:
            logger.info(f"Configured terminations: {list(cfg.terminations.keys())}")

    def _setup_spaces(self, robot_config: dict) -> None:
        """Set up action and observation spaces."""
        action_limits = robot_config["action_space"]["actuator_limits"]
        action_dim = len(action_limits)

        self.action_space = spaces.Box(
            low=np.array([limit["lower"] for limit in action_limits], dtype=np.float32),
            high=np.array([limit["upper"] for limit in action_limits], dtype=np.float32),
            shape=(action_dim,),
            dtype=np.float32,
        )

        obs_dim = self.observation_processor.output_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

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

    def _convert_observation(self, observation: ObservationResponse) -> np.ndarray:
        """Convert ObservationResponse to flat numpy array."""
        if observation.observation:
            obs = np.array(observation.observation, dtype=np.float32)
            expected_size = self.observation_space.shape[0]
            if len(obs) >= expected_size:
                obs = obs[:expected_size]
            else:
                padded = np.zeros(expected_size, dtype=np.float32)
                padded[: len(obs)] = obs
                obs = padded
            return np.clip(obs, self.observation_space.low, self.observation_space.high)
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Reset the environment.

        On reset, the extras["log"] dict is populated with episode statistics
        from all managers. This follows the mjlab pattern for logging.
        """
        super().reset(seed=seed, options=options)

        # Initialize log dict for this reset (mjlab pattern)
        self.extras["log"] = {}

        # Collect episode stats from managers before resetting
        if self.reward_manager is not None:
            info = self.reward_manager.reset()
            self.extras["log"].update(info)

        if self.termination_manager is not None:
            info = self.termination_manager.reset()
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
        if self.curriculum_manager is not None:
            self.curriculum_manager.update(self)

        # Reset episode state
        self.step_count = 0
        self.current_action = None
        self.last_action = None
        self.episode_metrics.reset()
        self.command_manager.reset()

        # Reset environment via gRPC
        observation = self.luckyrobots.reset()
        self.latest_observation = observation

        # Reset observation processor
        if observation.observation:
            raw_obs = np.array(observation.observation, dtype=np.float32)
            self.observation_processor.reset(raw_obs)
            self._obs_processor_initialized = True

        obs = self._convert_observation(observation)

        # Build info dict with command and log reference
        info = {
            "command": self.command_manager.get_command("velocity").tolist(),
            "log": self.extras["log"],
        }

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        self.step_count += 1
        self.common_step_counter += 1  # Track total steps for curriculum
        self.last_action = self.current_action.copy() if self.current_action is not None else np.zeros_like(action)
        self.current_action = action.copy()

        # Send control and get observation
        self.luckyrobots.send_control(controls=action.tolist())
        observation = self.luckyrobots.get_observation()
        self.latest_observation = observation

        # Update commands
        self.command_manager.update(self.cfg.sim_dt)

        # Build context for reward/termination functions
        context = self._build_mdp_context(observation)

        # Compute reward using reward manager
        info: dict = {"command": self.command_manager.get_command("velocity").tolist()}
        if self.reward_manager is not None:
            reward = self.reward_manager.compute(context)
            info["reward_details"] = {
                name: value for name, [value] in self.reward_manager.get_active_iterable_terms()
            }
        else:
            reward = self._compute_reward(context, info)

        # Compute termination using termination manager
        if self.termination_manager is not None:
            self.termination_manager.compute(context)
            terminated = self.termination_manager.terminated
            info["termination_reasons"] = self.termination_manager.termination_reasons
            info["termination_reason"] = (
                self.termination_manager.termination_reasons[0]
                if self.termination_manager.termination_reasons
                else "none"
            )
            info["is_success"] = False
        else:
            terminated = self._check_termination(context, info)

        # Apply observation processing (noise, delay, history)
        if observation.observation and self._obs_processor_initialized:
            raw_obs = np.array(observation.observation, dtype=np.float32)
            processed_obs = self.observation_processor.process(raw_obs)
            info["raw_observation"] = raw_obs
            info["processed_observation"] = processed_obs
            obs = processed_obs
        else:
            obs = self._convert_observation(observation)

        # Check truncation (timeout)
        truncated = self.step_count >= self.cfg.max_episode_length

        # Update episode metrics
        self.episode_metrics.add_step()
        self.episode_metrics.add_reward(reward)
        if terminated:
            self.episode_metrics.set_terminated(info.get("termination_reason", ""))
        if truncated:
            self.episode_metrics.set_truncated()

        return obs, reward, terminated, truncated, info

    def _build_mdp_context(self, observation: ObservationResponse) -> dict:
        """Build context dictionary for reward/termination functions.

        This provides all the data that reward/termination functions might need.
        """
        context = {
            "step_count": self.step_count,
            "max_steps": self.cfg.max_episode_length,
            "current_action": self.current_action,
            "last_action": self.last_action,
            "action_low": self.action_low,
            "action_high": self.action_high,
        }

        if observation.observation:
            obs_array = np.array(observation.observation, dtype=np.float32)
            obs_parsed = self.obs_parser.parse(obs_array)

            # Inject current command
            current_command = self.command_manager.get_command("velocity")
            obs_parsed["commands"] = current_command

            context["obs_parsed"] = obs_parsed
            context["obs_array"] = obs_array
        else:
            context["obs_parsed"] = None
            context["obs_array"] = None

        return context

    def _compute_reward(self, context: dict, info: dict) -> float:
        """Compute reward based on configuration.

        Iterates through all reward terms defined in cfg.rewards,
        calls each function directly, and computes weighted sum.
        """
        if context["obs_parsed"] is None:
            return 0.0

        total_reward = 0.0
        reward_details = {}

        for name, term in self.cfg.rewards.items():
            func = term.func
            kwargs = dict(term.params)

            # Handle special cases for different function signatures
            if func.__name__ == "action_rate_l2":
                if context["current_action"] is None:
                    continue
                r = func(context["current_action"], context["last_action"], **kwargs)
            elif func.__name__ == "joint_pos_limits":
                kwargs.setdefault("action_low", context["action_low"])
                kwargs.setdefault("action_high", context["action_high"])
                r = func(context["obs_parsed"], **kwargs)
            else:
                r = func(context["obs_parsed"], **kwargs)

            weighted_r = term.weight * r
            total_reward += weighted_r
            reward_details[name] = {"raw": r, "weighted": weighted_r}

        info["reward_details"] = reward_details
        return float(total_reward)

    def _check_termination(self, context: dict, info: dict) -> bool:
        """Check termination conditions based on configuration.

        Iterates through all termination terms defined in cfg.terminations,
        calls each function directly, and returns True if any condition is met.
        """
        info["is_success"] = False
        info["termination_reasons"] = []

        if context["obs_parsed"] is None:
            return False

        for name, term in self.cfg.terminations.items():
            func = term.func
            kwargs = dict(term.params)

            # Handle special cases for different function signatures
            if func.__name__ in ("time_out", "max_steps_termination"):
                result = func(context["step_count"], context["max_steps"], **kwargs)
            elif func.__name__ == "nan_detection":
                result = func(context["obs_parsed"])
            else:
                result = func(context["obs_parsed"], **kwargs)

            if result:
                info["termination_reasons"].append(name)

        terminated = len(info["termination_reasons"]) > 0
        info["termination_reason"] = info["termination_reasons"][0] if terminated else "none"

        return terminated

    def render(self) -> np.ndarray | None:
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return None

    def close(self) -> None:
        """Close the environment."""
        if self.luckyrobots:
            self.luckyrobots.close(stop_engine=not self.cfg.skip_launch)
        logger.info("Environment closed")

    def get_episode_log(self) -> dict:
        """Get the current episode's log data.

        Returns a dictionary of metrics collected during the last completed
        episode. This follows the mjlab pattern where extras["log"] contains
        aggregated metrics from all managers.

        Returns:
            Dictionary with keys like:
            - Episode_Reward/<term_name>: Cumulative reward for each term
            - Episode_Termination/<term_name>: Whether each termination triggered
            - Episode/total_reward: Total episode reward
            - Episode/length: Episode length in steps
            - Episode/terminated: Whether episode was terminated (failure)
            - Episode/truncated: Whether episode was truncated (timeout)
            - Episode/survived: Whether robot survived (truncated but not terminated)
        """
        return self.extras.get("log", {})

    def get_training_metrics(self) -> dict:
        """Get current training metrics for logging.

        Returns metrics useful for monitoring training progress, including
        cumulative statistics and current state. Uses task name as prefix.

        Returns:
            Dictionary with training metrics prefixed by task name.
        """
        prefix = self.task_name
        metrics = {
            f"{prefix}/total_steps": float(self.common_step_counter),
            f"{prefix}/total_episodes": float(self.episode_count),
            f"{prefix}/current_episode_length": float(self.step_count),
        }

        # Add current reward breakdown if available
        if self.reward_manager is not None:
            for name, value in self.reward_manager.episode_sums.items():
                metrics[f"{prefix}_Reward/{name}"] = value

        return metrics

    def log_metric(self, name: str, value: float) -> None:
        """Add a custom metric to the current episode's log.

        The metric will be prefixed with the task name automatically.

        Args:
            name: Metric name (will be prefixed with task name).
            value: Metric value.

        Example:
            >>> env.log_metric("command_tracking_error", 0.05)
            # Results in: "Locomotion/command_tracking_error": 0.05
        """
        key = f"{self.task_name}/{name}"
        self.extras["log"][key] = value

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Add multiple custom metrics to the current episode's log.

        All metrics will be prefixed with the task name automatically.

        Args:
            metrics: Dictionary of metric names to values.

        Example:
            >>> env.log_metrics({"vel_error": 0.1, "heading_error": 0.05})
            # Results in: "Locomotion/vel_error": 0.1, "Locomotion/heading_error": 0.05
        """
        for name, value in metrics.items():
            self.log_metric(name, value)
