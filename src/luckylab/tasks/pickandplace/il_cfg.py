"""IL configurations for pick-and-place tasks."""

from luckylab.il.config import IlRunnerCfg

SO100_PICKANDPLACE_ACT_CFG = IlRunnerCfg(
    policy="act",
    dataset_repo_id="luckyrobots/so100_pickandplace_sim",
    experiment_name="so100_pickandplace_act",
    robot="so100",
    scene="SO100 Pick And Place",
    task="pickandplace",
    host="localhost",
    port=50051,
)
