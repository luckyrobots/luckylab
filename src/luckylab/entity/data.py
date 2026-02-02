"""Entity data container for luckylab.

Matches mjlab's EntityData API but stores data from LuckyEngine observations.
All properties return tensors with shape (num_envs, dim).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import torch

logger = logging.getLogger(__name__)

# Type alias for setter functions
SetterFunc = Callable[["EntityData", torch.Tensor], None]


@dataclass
class ObservationSchema:
    """Schema defining how to parse observations from the agent.

    Maps observation names from the agent schema to EntityData properties.
    Each entry defines: observation_name -> (size, property_name)

    The schema is general and can be configured per-agent. Common observation
    types have standard sizes (velocities=3, quaternion=4, etc.) while
    joint-related observations use the num_joints from the entity.
    """

    # Mapping: schema_name -> (size_or_None, property_name)
    # If size is None, it will be inferred from num_joints
    mappings: dict[str, tuple[int | None, str]] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "ObservationSchema":
        """Create default schema with common observation mappings.

        This covers standard robot observation types. The schema names
        should match what the agent provides in its observation_names.
        """
        return cls(
            mappings={
                # Linear velocity (body frame) - various naming conventions
                "base_lin_vel": (3, "root_link_lin_vel_b"),
                "base_linear_velocity": (3, "root_link_lin_vel_b"),
                "linear_velocity": (3, "root_link_lin_vel_b"),
                "lin_vel": (3, "root_link_lin_vel_b"),
                # Angular velocity (body frame)
                "base_ang_vel": (3, "root_link_ang_vel_b"),
                "base_angular_velocity": (3, "root_link_ang_vel_b"),
                "angular_velocity": (3, "root_link_ang_vel_b"),
                "ang_vel": (3, "root_link_ang_vel_b"),
                # Quaternion (world frame)
                "base_quat": (4, "root_link_quat_w"),
                "quaternion": (4, "root_link_quat_w"),
                "orientation": (4, "root_link_quat_w"),
                # Projected gravity (body frame)
                "proj_grav": (3, "projected_gravity_b"),
                "projected_gravity": (3, "projected_gravity_b"),
                "gravity": (3, "projected_gravity_b"),
                # Joint positions (size = num_joints)
                "joint_pos": (None, "joint_pos"),
                "joint_positions": (None, "joint_pos"),
                # Joint velocities (size = num_joints)
                "joint_vel": (None, "joint_vel"),
                "joint_velocities": (None, "joint_vel"),
                # Privileged observations - singular names matching mjlab
                "foot_contact": (4, "foot_contact"),
                "foot_contacts": (4, "foot_contact"),  # Alias
                "foot_height": (4, "foot_height"),
                "foot_heights": (4, "foot_height"),  # Alias
                "foot_contact_forces": (4, "foot_contact_forces"),
                "foot_forces": (4, "foot_contact_forces"),  # Alias
                "foot_force": (4, "foot_contact_forces"),  # Alias
                "foot_air_time": (4, "foot_air_time"),
                "foot_air_times": (4, "foot_air_time"),  # Alias
            }
        )

    def get_size(self, name: str, num_joints: int) -> int:
        """Get the size of an observation by name.

        Args:
            name: Observation name from schema.
            num_joints: Number of joints (used for joint-related observations).

        Returns:
            Size of the observation, or 0 if unknown.
        """
        if name not in self.mappings:
            return 0
        size, _ = self.mappings[name]
        if size is None:
            return num_joints
        return size

    def get_property(self, name: str) -> str | None:
        """Get the EntityData property name for an observation.

        Args:
            name: Observation name from schema.

        Returns:
            Property name, or None if not mapped.
        """
        if name not in self.mappings:
            return None
        _, prop = self.mappings[name]
        return prop


@dataclass
class EntityData:
    """Data container for an entity.

    Stores robot state data received from LuckyEngine observations.
    Properties match mjlab's EntityData for compatibility with reward/termination functions.

    The observation parsing is driven by an ObservationSchema which maps
    agent observation names to EntityData properties. This makes it general
    to any agent schema, not specific to any task.
    """

    num_envs: int
    device: torch.device
    joint_names: list[str] = field(default_factory=list)
    observation_schema: ObservationSchema = field(default_factory=ObservationSchema.default)

    # Buffers (initialized in __post_init__)
    _root_link_lin_vel_b: torch.Tensor = field(init=False)
    _root_link_ang_vel_b: torch.Tensor = field(init=False)
    _root_link_quat_w: torch.Tensor = field(init=False)
    _projected_gravity_b: torch.Tensor = field(init=False)
    _joint_pos: torch.Tensor = field(init=False)
    _joint_vel: torch.Tensor = field(init=False)
    _default_joint_pos: torch.Tensor = field(init=False)
    _default_joint_vel: torch.Tensor = field(init=False)
    # Privileged observation buffers - singular names matching mjlab
    _foot_contact: torch.Tensor = field(init=False)
    _foot_height: torch.Tensor = field(init=False)
    _foot_contact_forces: torch.Tensor = field(init=False)
    _foot_air_time: torch.Tensor = field(init=False)

    def __post_init__(self):
        """Initialize data buffers."""
        self._root_link_lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._root_link_ang_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._root_link_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._root_link_quat_w[:, 0] = 1.0  # Default identity quaternion (w, x, y, z)
        self._projected_gravity_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._projected_gravity_b[:, 2] = -1.0  # Default upright orientation
        self._joint_pos = torch.zeros(self.num_envs, 0, device=self.device)
        self._joint_vel = torch.zeros(self.num_envs, 0, device=self.device)
        self._default_joint_pos = torch.zeros(self.num_envs, 0, device=self.device)
        self._default_joint_vel = torch.zeros(self.num_envs, 0, device=self.device)
        # Privileged observation buffers (4 feet for quadrupeds: FR, FL, RR, RL)
        self._foot_contact = torch.zeros(self.num_envs, 4, device=self.device)
        self._foot_height = torch.zeros(self.num_envs, 4, device=self.device)
        self._foot_contact_forces = torch.zeros(self.num_envs, 4, device=self.device)
        self._foot_air_time = torch.zeros(self.num_envs, 4, device=self.device)

        # Property name -> buffer mapping for dynamic updates
        self._property_buffers: dict[str, torch.Tensor] = {
            "root_link_lin_vel_b": self._root_link_lin_vel_b,
            "root_link_ang_vel_b": self._root_link_ang_vel_b,
            "root_link_quat_w": self._root_link_quat_w,
            "projected_gravity_b": self._projected_gravity_b,
            "joint_pos": self._joint_pos,
            "joint_vel": self._joint_vel,
            "foot_contact": self._foot_contact,
            "foot_height": self._foot_height,
            "foot_contact_forces": self._foot_contact_forces,
            "foot_air_time": self._foot_air_time,
        }

    def set_num_joints(self, num_joints: int, joint_names: list[str] | None = None) -> None:
        """Initialize joint buffers with correct size."""
        self._joint_pos = torch.zeros(self.num_envs, num_joints, device=self.device)
        self._joint_vel = torch.zeros(self.num_envs, num_joints, device=self.device)
        self._default_joint_pos = torch.zeros(self.num_envs, num_joints, device=self.device)
        self._default_joint_vel = torch.zeros(self.num_envs, num_joints, device=self.device)
        if joint_names is not None:
            self.joint_names = joint_names

        # Update property buffer mapping
        self._property_buffers["joint_pos"] = self._joint_pos
        self._property_buffers["joint_vel"] = self._joint_vel

    def set_default_joint_pos(self, default_positions: list[float] | torch.Tensor) -> None:
        """Set the default joint positions.

        Args:
            default_positions: Default joint positions, shape (num_joints,).
                Will be broadcast to all environments.
        """
        if isinstance(default_positions, list):
            default_positions = torch.tensor(default_positions, device=self.device, dtype=torch.float32)
        # Broadcast to all environments
        self._default_joint_pos[:] = default_positions.unsqueeze(0)

    # Properties matching mjlab's EntityData API

    @property
    def root_link_lin_vel_b(self) -> torch.Tensor:
        """Root link linear velocity in body frame. Shape (num_envs, 3)."""
        return self._root_link_lin_vel_b

    @property
    def root_link_ang_vel_b(self) -> torch.Tensor:
        """Root link angular velocity in body frame. Shape (num_envs, 3)."""
        return self._root_link_ang_vel_b

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        """Root link quaternion in world frame (w, x, y, z). Shape (num_envs, 4)."""
        return self._root_link_quat_w

    @property
    def projected_gravity_b(self) -> torch.Tensor:
        """Gravity vector projected into body frame. Shape (num_envs, 3)."""
        return self._projected_gravity_b

    @property
    def joint_pos(self) -> torch.Tensor:
        """Joint positions. Shape (num_envs, num_joints)."""
        return self._joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        """Joint velocities. Shape (num_envs, num_joints)."""
        return self._joint_vel

    @property
    def default_joint_pos(self) -> torch.Tensor:
        """Default joint positions. Shape (num_envs, num_joints)."""
        return self._default_joint_pos

    @property
    def default_joint_vel(self) -> torch.Tensor:
        """Default joint velocities. Shape (num_envs, num_joints)."""
        return self._default_joint_vel

    # Privileged observations - singular names matching mjlab

    @property
    def foot_contact(self) -> torch.Tensor:
        """Foot contact states (binary). Shape (num_envs, 4). Order: FR, FL, RR, RL."""
        return self._foot_contact

    @property
    def foot_height(self) -> torch.Tensor:
        """Foot heights above ground in meters. Shape (num_envs, 4). Order: FR, FL, RR, RL."""
        return self._foot_height

    @property
    def foot_contact_forces(self) -> torch.Tensor:
        """Foot contact force magnitudes in Newtons. Shape (num_envs, 4). Order: FR, FL, RR, RL."""
        return self._foot_contact_forces

    @property
    def foot_air_time(self) -> torch.Tensor:
        """Time since last foot contact in seconds. Shape (num_envs, 4). Order: FR, FL, RR, RL."""
        return self._foot_air_time

    # Aliases for compatibility with different naming conventions

    @property
    def base_lin_vel(self) -> torch.Tensor:
        """Alias for root_link_lin_vel_b."""
        return self._root_link_lin_vel_b

    @property
    def base_ang_vel(self) -> torch.Tensor:
        """Alias for root_link_ang_vel_b."""
        return self._root_link_ang_vel_b

    @property
    def projected_gravity(self) -> torch.Tensor:
        """Alias for projected_gravity_b."""
        return self._projected_gravity_b

    def update_from_observation(
        self,
        obs_tensor: torch.Tensor,
        observation_names: list[str],
    ) -> None:
        """Update data from flat observation tensor using the observation schema.

        Parses the flat observation tensor based on the observation_names from
        the agent schema. Each name is looked up in the ObservationSchema to
        determine its size and which EntityData property to update.

        Args:
            obs_tensor: Flat observation tensor from engine (obs_dim,) or (num_envs, obs_dim).
            observation_names: List of observation group names from agent schema.
        """
        # Ensure batch dimension
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        num_joints = self._joint_pos.shape[1] if self._joint_pos.numel() > 0 else 0

        idx = 0
        for name in observation_names:
            # Get size and property from schema
            size = self.observation_schema.get_size(name, num_joints)
            if size == 0:
                # Unknown observation - skip (or could log warning)
                continue

            prop_name = self.observation_schema.get_property(name)
            if prop_name is None:
                idx += size
                continue

            # Extract data slice
            data = obs_tensor[:, idx : idx + size]

            # Update the corresponding buffer
            buffer = self._property_buffers.get(prop_name)
            if buffer is not None and buffer.shape[1] == size:
                buffer[:] = data

            idx += size

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset data for specified environments."""
        self._root_link_lin_vel_b[env_ids] = 0.0
        self._root_link_ang_vel_b[env_ids] = 0.0
        self._root_link_quat_w[env_ids] = 0.0
        self._root_link_quat_w[env_ids, 0] = 1.0  # Identity quaternion
        self._projected_gravity_b[env_ids] = 0.0
        self._projected_gravity_b[env_ids, 2] = -1.0
        self._joint_pos[env_ids] = self._default_joint_pos[env_ids]
        self._joint_vel[env_ids] = 0.0

    def validate_observations(self, observation_names: list[str]) -> dict[str, bool]:
        """Check which observations are available from the engine.

        This is useful to log warnings about missing privileged observations
        (foot_contact, foot_height, etc.) that may be required by reward functions.

        Args:
            observation_names: List of observation names from the agent schema.

        Returns:
            Dict mapping observation property names to availability status.
        """
        # Check which properties we expect to be updated
        privileged_obs = {
            "foot_contact": False,
            "foot_height": False,
            "foot_contact_forces": False,
            "foot_air_time": False,
        }

        for name in observation_names:
            prop_name = self.observation_schema.get_property(name)
            if prop_name in privileged_obs:
                privileged_obs[prop_name] = True

        # Log warnings for missing privileged observations
        missing = [name for name, available in privileged_obs.items() if not available]
        if missing:
            logger.warning(
                "Privileged observations not provided by LuckyEngine: %s. "
                "Reward/termination functions using these will receive zero values. "
                "Available observation names from engine: %s",
                missing,
                observation_names,
            )

        return privileged_obs

    def has_foot_data(self) -> bool:
        """Check if any foot sensor data has been populated.

        Returns:
            True if any foot observation buffer has non-zero values.
        """
        return (
            self._foot_contact.any().item()
            or self._foot_height.any().item()
            or self._foot_contact_forces.any().item()
            or self._foot_air_time.any().item()
        )
