"""LuckyLab - RL and IL training framework for LuckyRobots."""

import logging
from importlib.metadata import entry_points
from pathlib import Path

logger = logging.getLogger(__name__)

LUCKYLAB_SRC_PATH: Path = Path(__file__).parent


def _import_registered_packages() -> None:
    """Auto-discover and import packages registered via entry points.

    Looks for packages registered under the 'luckylab.tasks' entry point group.
    Each discovered package is imported, which allows it to register custom
    environments with the task registry.
    """
    luckylab_tasks = entry_points().select(group="luckylab.tasks")
    for entry_point in luckylab_tasks:
        try:
            entry_point.load()
        except Exception as e:
            logger.warning("Failed to load task package %s: %s", entry_point.name, e)


_import_registered_packages()
