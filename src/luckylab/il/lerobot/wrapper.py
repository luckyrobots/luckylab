"""Thin gymnasium wrapper around luckyrobots.Session for LeRobot IL evaluation."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces


class LeRobotEnvWrapper(gymnasium.Env):
    """Wraps a luckyrobots.Session as a gymnasium.Env for LeRobot closed-loop eval.

    This is intentionally lighter than ManagerBasedRlEnv — IL eval doesn't need
    reward managers, termination managers, or curriculum. Returns dict observations
    as LeRobot policies expect.
    """

    metadata = {"render_modes": []}

    def __init__(self, session, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.session = session

        self.observation_space = spaces.Dict({
            "observation.state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
            ),
        })
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed, options=options)
        obs_response = self.session.reset()
        obs = {"observation.state": np.asarray(obs_response.observation, dtype=np.float32)}
        return obs, {}

    def step(
        self, action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        obs_response = self.session.step(actions=action.tolist())
        obs = {"observation.state": np.asarray(obs_response.observation, dtype=np.float32)}
        # IL eval is task-success based, not reward-based
        return obs, 0.0, False, False, {}

    def close(self) -> None:
        self.session.close(stop_engine=False)


def make_lerobot_env(
    host: str = "localhost",
    port: int = 8080,
    scene: str = "",
    robot: str = "",
    simulation_mode: str = "realtime",
) -> LeRobotEnvWrapper:
    """Factory to create a LeRobotEnvWrapper connected to LuckyEngine.

    Connects to an existing or new LuckyEngine instance, queries the agent schema
    for obs/action dimensions, and wraps as a gymnasium env.

    Args:
        host: LuckyEngine gRPC host.
        port: LuckyEngine gRPC port.
        scene: Scene name to load (if starting engine).
        robot: Robot type.
        simulation_mode: Timing mode ("realtime", "deterministic", "fast").

    Returns:
        A LeRobotEnvWrapper ready for policy evaluation.
    """
    from luckyrobots import Session

    session = Session(host=host, port=port)
    if scene:
        session.start(scene=scene, robot=robot, task="", headless=False)
    else:
        session.connect(robot=robot)

    session.set_simulation_mode(simulation_mode)

    # Query schema for obs/action dimensions
    schema_resp = session.engine_client.get_agent_schema()
    obs_dim = schema_resp.schema.observation_size
    action_dim = schema_resp.schema.action_size

    return LeRobotEnvWrapper(session, obs_dim=obs_dim, action_dim=action_dim)
