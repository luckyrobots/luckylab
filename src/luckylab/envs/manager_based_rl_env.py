import math
from dataclasses import dataclass, field
from typing import Any

from luckyrobots import LuckyRobots
import numpy as np
import torch
from prettytable import PrettyTable

from luckylab.envs import types
from luckylab.configs.domain_randomization import PhysicsDRCfg
from luckylab.entity import Entity, EntityCfg
from luckylab.scene import Scene
from luckylab.managers.action_manager import ActionManager
from luckylab.managers.command_manager import CommandManager, NullCommandManager
from luckylab.managers.curriculum_manager import CurriculumManager, NullCurriculumManager
from luckylab.managers.manager_term_config import (
    ActionTermCfg,
    CommandTermCfg,
    CurriculumTermCfg,
    ObservationGroupCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from luckylab.managers.observation_manager import ObservationManager
from luckylab.managers.reward_manager import RewardManager
from luckylab.managers.termination_manager import TerminationManager
from luckylab.utils import random as random_utils
from luckylab.utils.logging import print_info
from luckylab.utils.spaces import Box, Dict as DictSpace, batch_space


@dataclass(kw_only=True)
class ManagerBasedRlEnvCfg:
    """Configuration for a manager-based RL environment.

    Combines base environment settings with RL-specific configuration
    (rewards, terminations, commands, curriculum).
    """
    
    decimation: int
    """Number of simulation steps per environment step."""
    observations: dict[str, ObservationGroupCfg]
    """Observation groups: name -> group config."""
    actions: dict[str, ActionTermCfg]
    """Action terms: name -> config."""

    # Environment settings.
    seed: int | None = None
    """Random seed (None = random)."""
    num_envs: int = 1
    """Number of parallel environments."""
    episode_length_s: float = 20.0
    """Maximum episode length in seconds."""
    sim_dt: float = 0.02
    """Simulation timestep in seconds."""
    is_finite_horizon: bool = False
    """Whether the task has a finite horizon."""

    # LuckyRobots connection.
    scene: str = "velocity"
    """Scene name to load."""
    task: str = "locomotion"
    """Task name."""
    robot: str = "unitreego1"
    """Robot name."""
    host: str = "172.24.160.1"
    """gRPC host address."""
    port: int = 50051
    """gRPC port."""
    timeout_s: float = 120.0
    """Connection timeout in seconds."""
    skip_launch: bool = True
    """Skip launching engine (connect to existing)."""

    # MDP configs.
    rewards: dict[str, RewardTermCfg] = field(default_factory=dict)
    """Reward terms: name -> config."""
    terminations: dict[str, TerminationTermCfg] = field(default_factory=dict)
    """Termination terms: name -> config."""
    commands: dict[str, CommandTermCfg] | None = None
    """Command terms: name -> config."""
    curriculum: dict[str, CurriculumTermCfg] | None = None
    """Curriculum terms: name -> config."""
    is_finite_horizon: bool = False
    """Whether the task has a finite horizon."""
    physics_dr: PhysicsDRCfg = field(default_factory=PhysicsDRCfg)
    """Physics domain randomization config."""


class ManagerBasedRlEnv:
    """Manager-based RL environment for LuckyRobots.

    Connects to LuckyRobots via gRPC and provides a torch-native interface.
    All behavior is driven by manager configurations.
    """

    cfg: ManagerBasedRlEnvCfg

    def __init__(
            self, 
            cfg: ManagerBasedRlEnvCfg, 
            device: str = "cpu"
        ) -> None:
        """Initialize the environment.

        Args:
            cfg: Environment configuration
            device: Device for computation (e.g., "cuda:0")
            **kwargs: Additional arguments (unused, for compatibility)
        """
        # Initialize base environment state.
        self.cfg = cfg
        self._device = torch.device(device)
        self._num_envs = cfg.num_envs
        if self.cfg.seed is not None:
            self.cfg.seed = self.seed(self.cfg.seed)
        self.extras: dict[str, Any] = {}
        self.obs_buf: types.VecEnvObs = {}

        # Print environment info.
        print_info("")
        table = PrettyTable()
        table.title = "Base Environment"
        table.field_names = ["Property", "Value"]
        table.align["Property"] = "l"
        table.align["Value"] = "l"
        table.add_row(["Number of environments", self.num_envs])
        table.add_row(["Environment device", self.device])
        table.add_row(["Environment seed", self.cfg.seed])
        table.add_row(["Physics step-size", self.physics_dt])
        table.add_row(["Environment step-size", self.step_dt])
        print_info(table.get_string())
        print_info("")

        # Initialize RL-specific state.
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(
            self._num_envs, dtype=torch.long, device=self._device
        )

        # Connect to LuckyRobots.
        self.luckyrobots = LuckyRobots(host=cfg.host, port=cfg.port)
        self._connect()

        # Get robot configuration.
        robot_config = LuckyRobots.get_robot_config(cfg.robot)
        self._num_joints = len(robot_config["action_space"]["actuator_limits"])

        # Fetch observation schema from engine.
        self._observation_names: list[str] = []
        self._engine_obs_size: int = 0
        self._fetch_observation_schema()

        # Initialize scene with robot entity.
        self._init_scene(robot_config)

        self._load_managers()

        print_info(f"Environment initialized: {self._num_envs} envs, device={device}")

    # Properties.

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self._num_envs

    @property
    def device(self) -> torch.device:
        """Device for computation."""
        return self._device

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        return self.cfg.episode_length_s
    
    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in steps."""
        return math.ceil(self.max_episode_length_s / self.step_dt)

    @property
    def physics_dt(self) -> float:
        """Physics simulation step size."""
        return self.cfg.sim_dt

    @property
    def step_dt(self) -> float:
        """Environment step size (physics_dt * decimation)."""
        return self.physics_dt * self.cfg.decimation

    @property
    def unwrapped(self) -> "ManagerBasedRlEnv":
        """Return the unwrapped environment."""
        return self

    # Methods.

    def _load_managers(self) -> None:
        """Load and initialize all managers.

        Order is important! Event and command managers must be loaded first,
        then action and observation managers, then other RL managers.
        """
        cfg = self.cfg

        # Command manager (must be before observation manager since observations may reference commands).
        if cfg.commands:
            self.command_manager = CommandManager(cfg.commands, self)
        else:
            self.command_manager = NullCommandManager()
        print_info(f"[INFO] {self.command_manager}")

        # Action and observation managers.
        self.action_manager = ActionManager(cfg.actions, self)
        print_info(f"[INFO] {self.action_manager}")
        self.observation_manager = ObservationManager(cfg.observations, self)
        print_info(f"[INFO] {self.observation_manager}")

        # Other RL-specific managers.

        self.termination_manager = TerminationManager(cfg.terminations, self)
        print_info(f"[INFO] {self.termination_manager}")
        self.reward_manager = RewardManager(cfg.rewards, self)
        print_info(f"[INFO] {self.reward_manager}")
        # Curriculum manager.
        if cfg.curriculum:
            self.curriculum_manager = CurriculumManager(cfg.curriculum, self)
        else:
            self.curriculum_manager = NullCurriculumManager()
        print_info(f"[INFO] {self.curriculum_manager}")

        self._configure_gym_env_spaces()

    def reset(
        self,
        seed: int | None = None,
        env_ids: torch.Tensor | None = None,
        options: dict | None = None,
    ) -> tuple[types.VecEnvObs, dict]:
        """Reset the environment.

        Args:
            seed: Random seed.
            env_ids: Environment indices to reset (default: all).
            options: Additional options (unused).

        Returns:
            Tuple of (observations dict, extras).
        """
        del options

        if env_ids is None:
            env_ids = torch.arange(self._num_envs, dtype=torch.long, device=self._device)

        if seed is not None:
            self.seed(seed)

        self._reset_idx(env_ids)

        # Compute all observation groups.
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.extras

    def step(self, action: torch.Tensor) -> types.VecEnvStepReturn:
        """Execute one environment step.

        Args:
            action: Action tensor of shape (num_envs, action_dim).

        Returns:
            Tuple of (observations dict, reward, terminated, truncated, extras).
        """
        if action.dim() == 1:
            action = action.unsqueeze(0)

        # Process action through action manager (updates action history internally).
        self.action_manager.process_action(action)

        # Update counters.
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # Check terminations.
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # Get processed joint positions and step physics via gRPC.
        joint_positions = self.action_manager.processed_action
        joint_list = joint_positions[0].detach().cpu().tolist()
        observation = self.luckyrobots.step(actions=joint_list)

        # Update entity data from observation.
        if observation.observation:
            obs_tensor = torch.tensor(
                observation.observation, dtype=torch.float32, device=self._device
            )
            self.scene["robot"].data.update_from_observation(obs_tensor, self._observation_names)

        # Update commands.
        self.command_manager.compute(dt=self.step_dt)

        self.obs_buf = self.observation_manager.compute(update_history=True)

        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )
    
    def render(self) -> torch.Tensor | None:
        """Render the environment."""
        return None

    def close(self) -> None:
        """Close the environment."""
        if self.luckyrobots:
            self.luckyrobots.close(stop_engine=not self.cfg.skip_launch)
        print_info("Environment closed")

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Set the random seed.

        Args:
            seed: Seed value (-1 = random)

        Returns:
            The actual seed used
        """
        if seed == -1:
            seed = np.random.randint(0, 10_000)
        print_info(f"Setting seed: {seed}")
        random_utils.seed_rng(seed)
        return seed

    # Private methods.

    def _connect(self) -> None:
        """Connect to LuckyRobots engine."""
        if self.cfg.skip_launch:
            print_info(f"Connecting to LuckyEngine at {self.cfg.host}:{self.cfg.port}")
            self.luckyrobots.connect(timeout_s=self.cfg.timeout_s, robot=self.cfg.robot)
        else:
            print_info("Launching LuckyEngine...")
            self.luckyrobots.start(
                scene=self.cfg.scene,
                robot=self.cfg.robot,
                task=self.cfg.task,
                timeout_s=self.cfg.timeout_s,
            )
        self.luckyrobots.set_simulation_mode("fast")

    def _fetch_observation_schema(self) -> None:
        """Fetch observation schema from engine."""
        if self.luckyrobots.engine_client is None:
            raise RuntimeError("Engine client not connected")

        schema_resp = self.luckyrobots.engine_client.get_agent_schema()
        if not hasattr(schema_resp, "schema") or not schema_resp.schema:
            raise RuntimeError("Engine returned empty schema")

        self._observation_names = list(schema_resp.schema.observation_names)
        self._engine_obs_size = schema_resp.schema.observation_size
        print_info(f"Observation schema: {len(self._observation_names)} fields, size={self._engine_obs_size}")

    def _init_scene(self, robot_config: dict) -> None:
        """Initialize scene with robot entity."""
        action_limits = robot_config["action_space"]["actuator_limits"]
        joint_names = [limit.get("name", f"joint_{i}") for i, limit in enumerate(action_limits)]

        action_space = robot_config["action_space"]
        if "actuator_names" in action_space:
            actuator_names = list(action_space["actuator_names"])
        else:
            actuator_names = joint_names

        self.scene = Scene()
        robot = Entity(
            cfg=EntityCfg(),
            num_envs=self._num_envs,
            num_joints=self._num_joints,
            joint_names=joint_names,
            actuator_names=actuator_names,
            device=self._device,
        )
        self.scene.add("robot", robot)

        # Set default joint positions.
        default_positions = [limit.get("default", 0.0) for limit in action_limits]
        robot.data.set_default_joint_pos(default_positions)

        # Validate available observations.
        if self._observation_names:
            robot.data.validate_observations(self._observation_names)

    def _configure_gym_env_spaces(self) -> None:
        """Configure observation and action spaces.

        Builds a DictSpace containing all observation groups.
        Each group is either a Box (concatenated) or nested DictSpace (non-concatenated).
        """
        # Build observation space as DictSpace with all groups.
        self.single_observation_space = DictSpace()
        for group_name, group_term_names in self.observation_manager.active_terms.items():
            has_concatenated_obs = self.observation_manager.group_obs_concatenate[group_name]
            group_dim = self.observation_manager.group_obs_dim[group_name]

            if has_concatenated_obs:
                # Concatenated group: single Box with total dimension.
                assert isinstance(group_dim, tuple), f"Expected tuple for concatenated group {group_name}"
                self.single_observation_space.spaces[group_name] = Box(
                    shape=group_dim, low=-math.inf, high=math.inf
                )
            else:
                # Non-concatenated group: nested DictSpace with per-term boxes.
                assert not isinstance(group_dim, tuple), f"Expected list for non-concatenated group {group_name}"
                group_term_cfgs = self.observation_manager._group_obs_term_cfgs[group_name]
                group_space = DictSpace()
                for term_name, term_dim, _term_cfg in zip(
                    group_term_names, group_dim, group_term_cfgs, strict=False
                ):
                    group_space.spaces[term_name] = Box(
                        shape=term_dim, low=-math.inf, high=math.inf
                    )
                self.single_observation_space.spaces[group_name] = group_space

        # Action space.
        action_dim = sum(self.action_manager.action_term_dim)
        self.single_action_space = Box(shape=(action_dim,), low=-math.inf, high=math.inf)

        # Batched spaces.
        self.observation_space = batch_space(self.single_observation_space, self._num_envs)
        self.action_space = batch_space(self.single_action_space, self._num_envs)

        print_info(f"Spaces: obs_groups={list(self.single_observation_space.spaces.keys())}, action={action_dim}")

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset specific environments.

        Args:
            env_ids: Environment indices to reset.
        """
        # Log episode length before reset.
        if len(env_ids) > 0:
            self.extras["log"]["Episode/length"] = torch.mean(
                self.episode_length_buf[env_ids].float()
            ).item()

        # Update curriculum.
        self.curriculum_manager.compute(env_ids)

        # NOTE: This is order sensitive.
        self.extras["log"] = dict()
        # observation manager.
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # action manager.
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # rewards manager.
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        # curriculum manager.
        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)
        # command manager.
        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)
        # termination manager.
        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)
        # reset the episode length buffer.
        self.episode_length_buf[env_ids] = 0

        # Reset physics via gRPC.
        observation = self.luckyrobots.reset(randomization_cfg=self.cfg.physics_dr)

        # Update entity data.
        if observation.observation:
            obs_tensor = torch.tensor(
                observation.observation, dtype=torch.float32, device=self._device
            )
            self.scene["robot"].data.update_from_observation(obs_tensor, self._observation_names)
