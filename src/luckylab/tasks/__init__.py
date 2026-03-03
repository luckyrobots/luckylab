"""Task configurations for luckylab."""

from luckylab.tasks.registry import (
    list_il_policies as list_il_policies,
)
from luckylab.tasks.registry import (
    list_rl_policies as list_rl_policies,
)
from luckylab.tasks.registry import (
    list_tasks as list_tasks,
)
from luckylab.tasks.registry import (
    load_env_cfg as load_env_cfg,
)
from luckylab.tasks.registry import (
    load_il_cfg as load_il_cfg,
)
from luckylab.tasks.registry import (
    load_rl_cfg as load_rl_cfg,
)
from luckylab.tasks.registry import (
    register_task as register_task,
)
from luckylab.utils.importer import import_packages

# Auto-discover and import all task packages
_BLACKLIST_PKGS = ["utils", "registry", ".mdp"]
import_packages(__name__, _BLACKLIST_PKGS)
