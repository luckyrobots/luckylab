"""LuckyLab - RL training framework for LuckyRobots."""

from importlib.metadata import entry_points
from pathlib import Path

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
            print(f"[WARN] Failed to load task package {entry_point.name}: {e}")


_import_registered_packages()
