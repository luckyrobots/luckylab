"""Piper block-stacking IL task (no env config — offline dataset only)."""

from luckylab.il.config import IlRunnerCfg
from luckylab.tasks.registry import register_task

register_task(
    task_id="piper_blockstacking",
    il_cfgs={
        "act": IlRunnerCfg(
            policy="act",
            dataset_repo_id="piper/blockstacking",
            experiment_name="piper_blockstacking_act",
            robot="piper",
            scene="Piper-Block-Stacking",
            host="localhost",
            port=50051,
            skip_launch=False,
        ),
        "diffusion": IlRunnerCfg(
            policy="diffusion",
            dataset_repo_id="piper/blockstacking",
            experiment_name="piper_blockstacking_diffusion",
            robot="piper",
            scene="Piper-Block-Stacking",
            host="localhost",
            port=50051,
            skip_launch=False,
        ),
    },
)
