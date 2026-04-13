"""Pick-and-place IL tasks.

Each robot gets its own register_task() call with a unique task_id.
To add a new robot, define configs in il_cfg.py and register here.
"""

from luckylab.tasks.registry import register_task

from .il_cfg import SO100_PICKANDPLACE_ACT_CFG

register_task(
    task_id="so100_pickandplace",
    il_cfgs={
        "act": SO100_PICKANDPLACE_ACT_CFG,
    },
)
