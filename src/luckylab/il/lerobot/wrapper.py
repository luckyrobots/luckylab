"""Thin gymnasium wrapper around luckyrobots.Session for LeRobot IL evaluation."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from luckylab.il.config import IlRunnerCfg
from luckylab.utils.logging import print_info


class LeRobotEnvWrapper(gymnasium.Env):
    """Wraps a luckyrobots.Session as a gymnasium.Env for LeRobot closed-loop eval.

    This is intentionally lighter than ManagerBasedRlEnv — IL eval doesn't need
    reward managers, termination managers, or curriculum. Returns dict observations
    as LeRobot policies expect.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        session,
        obs_dim: int,
        action_dim: int,
        camera_names: list[str] | None = None,
        camera_width: int = 256,
        camera_height: int = 256,
    ) -> None:
        super().__init__()
        self.session = session
        self._skip_launch = True
        self._camera_names = camera_names or []
        self._action_dim = action_dim
        self._obs_dim = obs_dim

        obs_spaces: dict[str, spaces.Space] = {
            "observation.state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
            ),
        }
        for cam_name in self._camera_names:
            obs_spaces[f"observation.images.{cam_name}"] = spaces.Box(
                low=0, high=255,
                shape=(3, camera_height, camera_width),
                dtype=np.float32,
            )

        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32,
        )

    def _parse_obs(self, obs_response) -> dict[str, np.ndarray]:
        """Parse an ObservationResponse into the dict format LeRobot expects."""
        state = np.asarray(obs_response.observation, dtype=np.float32)
        # Engine may return more values than the policy expects (e.g. pos+vel vs pos only).
        # Slice to the expected state dimension.
        obs: dict[str, np.ndarray] = {
            "observation.state": state[:self._obs_dim],
        }
        for cf in obs_response.camera_frames:
            pixels = np.frombuffer(cf.data, dtype=np.uint8).reshape(
                cf.height, cf.width, cf.channels,
            )
            # RGBA -> RGB, HWC -> CHW, uint8 -> float32
            rgb = pixels[:, :, :3].transpose(2, 0, 1).astype(np.float32)
            obs[f"observation.images.{cf.name}"] = rgb
        return obs

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed, options=options)
        obs_response = self.session.reset()
        # Pass through reset message from engine (contains episode eval results)
        info: dict[str, Any] = {}
        msg = getattr(self.session, "last_reset_message", "")
        if msg and "episode_success=" in msg:
            info["prev_episode_success"] = "True" in msg
        return self._parse_obs(obs_response), info

    def step(
        self, action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        obs_response = self.session.step(actions=action.tolist())
        return self._parse_obs(obs_response), 0.0, False, False, {}

    def close(self) -> None:
        self.session.close(stop_engine=not self._skip_launch)


def make_lerobot_env(
    il_cfg: IlRunnerCfg,
    obs_dim: int,
    action_dim: int,
    camera_names: list[str] | None = None,
    camera_width: int = 256,
    camera_height: int = 256,
) -> LeRobotEnvWrapper:
    """Factory to create a LeRobotEnvWrapper connected to LuckyEngine.

    Follows the same connection pattern as ManagerBasedRlEnv: skip_launch=True
    connects to an already-running engine, skip_launch=False launches one first.

    Args:
        il_cfg: IL runner configuration with connection settings.
        obs_dim: State observation dimension expected by the policy.
        action_dim: Action dimension expected by the policy.
        camera_names: Camera names to request from engine (e.g. ["Camera"]).
        camera_width: Requested camera image width.
        camera_height: Requested camera image height.

    Returns:
        A LeRobotEnvWrapper ready for policy evaluation.
    """
    from luckyrobots import Session

    session = Session(host=il_cfg.host, port=il_cfg.port)

    if il_cfg.skip_launch:
        print_info(f"Connecting to LuckyEngine at {il_cfg.host}:{il_cfg.port}")
        session.connect(timeout_s=il_cfg.timeout_s, robot=il_cfg.robot)
    else:
        print_info("Launching LuckyEngine...")
        session.start(
            scene=il_cfg.scene,
            robot=il_cfg.robot,
            task="",
            timeout_s=il_cfg.timeout_s,
        )

    # Override the default 5s per-RPC timeout on the underlying gRPC client.
    if session.engine_client is not None:
        session.engine_client.timeout = il_cfg.step_timeout_s

    # Configure cameras to capture on every step.
    if camera_names:
        session.configure_cameras([
            {"name": name, "width": camera_width, "height": camera_height}
            for name in camera_names
        ])

    env = LeRobotEnvWrapper(
        session,
        obs_dim=obs_dim,
        action_dim=action_dim,
        camera_names=camera_names,
        camera_width=camera_width,
        camera_height=camera_height,
    )
    env._skip_launch = il_cfg.skip_launch
    return env
