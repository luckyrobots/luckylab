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
    ) -> None:
        self.cfg = cfg
        self._num_envs = num_envs
        self._num_joints = num_joints
        self._joint_names = tuple(joint_names)
        self._device = device

        # Initialize data container
        self._data = EntityData(
            num_envs=num_envs,
            device=device,
            joint_names=joint_names,
        )
        self._data.set_num_joints(num_joints, joint_names)

    @property
    def data(self) -> EntityData:
        """Entity data container."""
        return self._data

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Joint names."""
        return self._joint_names

    @property
    def num_joints(self) -> int:
        """Number of joints."""
        return self._num_joints

    def find_joints(
        self,
        name_keys: str | Sequence[str],
        joint_subset: Sequence[str] | None = None,
    ) -> tuple[list[int], list[str]]:
        """Find joints matching the given name patterns.

        Args:
            name_keys: Regex pattern(s) to match joint names.
            joint_subset: Optional subset of joints to search.

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
            indices, names, _ = resolve_matching_names_values(data, list(joint_subset))
            return indices, names
        except ValueError:
            return [], []

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset entity state for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        self._data.reset(env_ids)
