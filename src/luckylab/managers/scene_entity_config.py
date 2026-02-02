"""Configuration for scene entities used by manager terms.

Matches mjlab/managers/scene_entity_config.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..entity import Entity, Scene


@dataclass
class SceneEntityCfg:
    """Configuration for a scene entity that is used by manager terms.

    Matches mjlab's SceneEntityCfg API. Not all fields are used by luckylab
    since we don't have access to body/geom/site level data from LuckyEngine,
    but the API is kept consistent for compatibility.

    Attributes:
        name: The name of the entity in the scene.
        joint_names: Names of joints to include.
        joint_ids: IDs of joints to include.
        body_names: Names of bodies to include (for API compatibility).
        body_ids: IDs of bodies to include (for API compatibility).
        preserve_order: Whether to preserve the order of names.
    """

    name: str

    joint_names: str | tuple[str, ...] | None = None
    joint_ids: list[int] | slice = field(default_factory=lambda: slice(None))

    body_names: str | tuple[str, ...] | None = None
    body_ids: list[int] | slice = field(default_factory=lambda: slice(None))

    preserve_order: bool = False

    def resolve(self, scene: "Scene") -> None:
        """Resolve names and IDs for configured fields.

        Args:
            scene: The scene containing the entity to resolve against.
        """
        entity = scene[self.name]

        # Resolve joint names to IDs
        if self.joint_names is not None:
            # Normalize to list
            if isinstance(self.joint_names, str):
                self.joint_names = (self.joint_names,)

            # Find matching joints
            indices, _ = entity.find_joints(self.joint_names)
            if indices:
                self.joint_ids = indices

        # Note: body_names/body_ids are kept for API compatibility but
        # luckylab doesn't resolve them since we don't have body-level data
        # from LuckyEngine. The root body data is always used.
