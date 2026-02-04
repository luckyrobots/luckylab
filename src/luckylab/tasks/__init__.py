"""Task configurations for luckylab."""

from luckylab.tasks.registry import (
    list_algorithms as list_algorithms,
    list_tasks as list_tasks,
    load_env_cfg as load_env_cfg,
    load_rl_cfg as load_rl_cfg,
    register_task as register_task,
)
from luckylab.utils.importer import import_packages

# Auto-discover and import all task packages
_BLACKLIST_PKGS = ["utils", "registry", ".mdp"]
import_packages(__name__, _BLACKLIST_PKGS)
