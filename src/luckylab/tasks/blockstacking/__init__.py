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
            scene="blockstacking",
            host="172.24.160.1",
            port=50051,
        ),
        "diffusion": IlRunnerCfg(
            policy="diffusion",
            dataset_repo_id="piper/blockstacking",
            experiment_name="piper_blockstacking_diffusion",
            robot="piper",
            scene="blockstacking",
            host="172.24.160.1",
            port=50051,
        ),
    },
)
