"""Scene class for luckylab.

Matches mjlab's scene access pattern: env.scene["robot"].data.*
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from luckylab.entity import Entity


class Scene:
    """Container for entities in the simulation.

    Provides dict-like access to entities: scene["robot"] returns the robot entity.
    """

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}

    def __getitem__(self, name: str) -> "Entity":
        """Get entity by name."""
        if name not in self._entities:
            raise KeyError(f"Entity '{name}' not found in scene. Available: {list(self._entities.keys())}")
        return self._entities[name]

    def __contains__(self, name: str) -> bool:
        """Check if entity exists."""
        return name in self._entities

    def add(self, name: str, entity: "Entity") -> None:
        """Add entity to scene."""
        self._entities[name] = entity

    def get(self, name: str, default: "Entity | None" = None) -> "Entity | None":
        """Get entity by name with default."""
        return self._entities.get(name, default)

    @property
    def entities(self) -> dict[str, "Entity"]:
        """All entities in the scene."""
        return self._entities
