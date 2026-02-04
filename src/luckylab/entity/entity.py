"""Entity class for luckylab.

Matches mjlab's Entity API but works with LuckyEngine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from luckylab.utils.string import resolve_matching_names_values

from .data import EntityData


@dataclass
class ActuatorInfo:
    """Information about an actuator.

    In luckylab, actuators are defined by LuckyEngine. This class stores
    the mapping between actuator names and the joints they control.
    """

    name: str
    """Actuator name (from LuckyEngine config)."""
    joint_name: str
    """Name of the joint this actuator controls."""
    joint_id: int
    """Index of the joint in the joint array."""


@dataclass
class EntityCfg:
    """Configuration for an entity."""

    @dataclass
    class InitialStateCfg:
        joint_pos: dict[str, float] = field(default_factory=lambda: {".*": 0.0})
        joint_vel: dict[str, float] = field(default_factory=lambda: {".*": 0.0})

    init_state: InitialStateCfg = field(default_factory=InitialStateCfg)


class Entity:
    """An entity represents a physical object (robot) in the simulation.

    Simplified version of mjlab's Entity that works with LuckyEngine.
    Physics are handled by LuckyEngine - this class just stores observation data.
    """

    def __init__(
        self,
        cfg: EntityCfg,
        num_envs: int,
        num_joints: int,
        joint_names: list[str],
        device: torch.device,
        actuator_names: list[str] | None = None,
    ) -> None:
        """Initialize the entity.

        Args:
            cfg: Entity configuration.
            num_envs: Number of parallel environments.
            num_joints: Number of joints.
            joint_names: List of joint names.
            device: Torch device.
            actuator_names: List of actuator names. If None, assumes actuator
                names match joint names (1:1 mapping).
        """
        self.cfg = cfg
        self._num_envs = num_envs
        self._num_joints = num_joints
        self._joint_names = tuple(joint_names)
        self._device = device

        # Initialize actuators
        # Default: assume actuator names = joint names (common case)
        if actuator_names is None:
            actuator_names = joint_names

        self._actuator_names = tuple(actuator_names)
        self._actuators: list[ActuatorInfo] = []
        self._actuator_to_joint: dict[str, str] = {}

        # Build actuator info list
        # For now, assume 1:1 mapping where actuator name = joint name
        # This can be extended later if LuckyEngine supports different mappings
        for i, act_name in enumerate(actuator_names):
            # Find corresponding joint (by name match or index)
            if act_name in joint_names:
                joint_name = act_name
                joint_id = joint_names.index(act_name)
            elif i < len(joint_names):
                # Fall back to index-based mapping
                joint_name = joint_names[i]
                joint_id = i
            else:
                raise ValueError(
                    f"Cannot map actuator '{act_name}' to a joint. "
                    f"Available joints: {joint_names}"
                )

            self._actuators.append(
                ActuatorInfo(name=act_name, joint_name=joint_name, joint_id=joint_id)
            )
            self._actuator_to_joint[act_name] = joint_name

        # Initialize data container
        self._data = EntityData(
            num_envs=num_envs,
            device=device,
            num_joints=num_joints,
            joint_names=joint_names,
        )

    @property
    def data(self) -> EntityData:
        """Entity data container."""
        return self._data
    
    @property
    def actuators(self) -> list[ActuatorInfo]:
        """List of actuator info objects."""
        return self._actuators

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Joint names."""
        return self._joint_names
    
    @property
    def actuator_names(self) -> tuple[str, ...]:
        """Actuator names."""
        return self._actuator_names

    @property
    def num_joints(self) -> int:
        """Number of joints."""
        return self._num_joints

    @property
    def num_actuators(self) -> int:
        """Number of actuators."""
        return len(self._actuators)

    def find_joints(
        self,
        name_keys: str | Sequence[str],
        joint_subset: Sequence[str] | None = None,
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        """Find joints matching the given name patterns.

        Args:
            name_keys: Regex pattern(s) to match joint names.
            joint_subset: Optional subset of joints to search.
            preserve_order: Whether to preserve the order of name_keys in output.

        Returns:
            Tuple of (indices, names) for matching joints.
        """
        if joint_subset is None:
            joint_subset = self._joint_names

        if isinstance(name_keys, str):
            name_keys = [name_keys]

        # Use resolve_matching_names_values with dummy values
        data = {key: 0 for key in name_keys}
        try:
            indices, names, _ = resolve_matching_names_values(data, list(joint_subset), preserve_order)
            return indices, names
        except ValueError:
            return [], []

    def find_actuators(
        self,
        name_keys: str | Sequence[str],
        actuator_subset: Sequence[str] | None = None,
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        """Find actuators matching the given name patterns.

        Args:
            name_keys: Regex pattern(s) to match actuator names.
            actuator_subset: Optional subset of actuators to search.
            preserve_order: Whether to preserve the order of name_keys in output.

        Returns:
            Tuple of (indices, names) for matching actuators.
        """
        if actuator_subset is None:
            actuator_subset = self._actuator_names

        if isinstance(name_keys, str):
            name_keys = [name_keys]

        data = {key: 0 for key in name_keys}
        try:
            indices, names, _ = resolve_matching_names_values(data, list(actuator_subset), preserve_order)
            return indices, names
        except ValueError:
            return [], []

    def find_joints_by_actuator_names(
        self,
        actuator_name_keys: str | Sequence[str],
    ) -> tuple[list[int], list[str]]:
        """Find actuated joints matching the given actuator name patterns.

        This method matches patterns against actuator names and returns the
        corresponding joint indices and names in natural joint order.

        Args:
            actuator_name_keys: Regex pattern(s) to match actuator names.

        Returns:
            Tuple of (joint_indices, joint_names) for joints controlled by
            matching actuators, in natural joint order.
        """
        # Find matching actuators
        _, matched_actuator_names = self.find_actuators(actuator_name_keys)

        if not matched_actuator_names:
            return [], []

        # Collect joint names for matched actuators
        actuated_joint_names_set = set()
        for act_name in matched_actuator_names:
            if act_name in self._actuator_to_joint:
                actuated_joint_names_set.add(self._actuator_to_joint[act_name])

        # Filter joint names to only actuated joints, preserving natural order
        actuated_in_natural_order = [
            name for name in self._joint_names if name in actuated_joint_names_set
        ]

        # Get indices for these joints
        joint_indices = [
            self._joint_names.index(name) for name in actuated_in_natural_order
        ]

        return joint_indices, actuated_in_natural_order

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset entity state for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        self._data.reset(env_ids)
