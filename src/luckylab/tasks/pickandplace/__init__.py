"""Piper pick-and-place IL task (no env config — offline dataset only)."""

from luckylab.il.config import IlRunnerCfg
from luckylab.tasks.registry import register_task

register_task(
    task_id="piper_pickandplace",
    il_cfgs={
        "act": IlRunnerCfg(
            policy="act",
            dataset_repo_id="piper/pickandplace",
            experiment_name="piper_pickandplace_act",
        ),
        "diffusion": IlRunnerCfg(
            policy="diffusion",
            dataset_repo_id="piper/pickandplace",
            experiment_name="piper_pickandplace_diffusion",
        ),
    },
)
