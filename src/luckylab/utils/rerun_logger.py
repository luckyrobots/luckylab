"""Rerun visualization logger for LuckyLab.

Provides live monitoring and episode-level inspection for RL and IL workflows.
Complements wandb (aggregate training metrics) with per-step visualization.

Usage:
    # RL: attached to env, logs automatically in step()
    env.rerun_logger = RerunLogger("luckylab/training")

    # IL: explicit calls in eval loop
    with RerunLogger("luckylab/il_eval") as rr_log:
        rr_log.log_il_step(obs, action, step=i)

    # Dataset visualization
    with RerunLogger("luckylab/dataset") as rr_log:
        rr_log.log_observation(obs, names=["x", "y", "z"])
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from urllib.parse import quote

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from luckylab.entity.data import EntityData
    from luckylab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from luckylab.managers.action_manager import ActionManager
    from luckylab.managers.reward_manager import RewardManager
    from luckylab.managers.termination_manager import TerminationManager


_FOOT_NAMES = ("FL", "FR", "RL", "RR")


class RerunLogger:
    """Rerun visualization logger with RL and IL convenience methods.

    Lazy-imports rerun-sdk so there is zero import cost when not used.
    Supports context manager protocol for clean resource management.
    """

    def __init__(
        self,
        app_id: str = "luckylab",
        save_path: str | None = None,
        log_interval: int = 1,
        env_idx: int = 0,
        web: bool = False,
        port: int = 9090,
    ) -> None:
        """Initialize the rerun logger.

        Args:
            app_id: Rerun application ID (shown in viewer title).
            save_path: If set, save to .rrd file instead of spawning viewer.
            log_interval: Only log every Nth call to on_rl_step (1 = every step).
            env_idx: Which environment index to log for vectorized envs.
            web: If True, serve a web viewer instead of spawning native viewer.
            port: Port for the web viewer (default 9090).
        """
        import rerun as rr

        self._rr = rr
        self._log_interval = max(1, log_interval)
        self._env_idx = env_idx
        self._call_count = 0
        self._web_url: str | None = None

        rr.init(app_id)
        if save_path is not None:
            rr.save(save_path)
        elif web:
            grpc_port = port + 1
            rr.serve_grpc(grpc_port=grpc_port)
            rr.serve_web_viewer(open_browser=False, web_port=port)
            # Build the full URL with ?url= param so the viewer auto-connects
            grpc_uri = f"rerun+http://localhost:{grpc_port}/proxy"
            self._web_url = f"http://localhost:{port}?url={quote(grpc_uri, safe='')}"
            logger.info("Rerun web viewer: %s", self._web_url)
        else:
            rr.spawn()

    # -- Context manager --

    def __enter__(self) -> RerunLogger:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # -- Generic methods --

    def set_step(self, step: int) -> None:
        """Set the current time step for all subsequent logs."""
        self._rr.set_time("step", sequence=step)

    def log_scalar(self, path: str, value: float) -> None:
        """Log a scalar value."""
        self._rr.log(path, self._rr.Scalars(value))

    def log_image(self, path: str, image: np.ndarray) -> None:
        """Log an image (HWC or CHW numpy array)."""
        if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[2] not in (1, 3, 4):
            # CHW -> HWC
            image = np.transpose(image, (1, 2, 0))
        self._rr.log(path, self._rr.Image(image))

    def log_text(self, path: str, text: str) -> None:
        """Log a text entry."""
        self._rr.log(path, self._rr.TextLog(text))

    def close(self) -> None:
        """Finalize the rerun recording."""
        # rerun-sdk doesn't require explicit close, but this is here
        # for forward compatibility and context manager support.
        pass

    # -- RL convenience methods --

    def log_entity_data(
        self,
        data: EntityData,
        joint_names: list[str],
        env_idx: int | None = None,
    ) -> None:
        """Log entity state data (joints, velocity, gravity, feet, commands).

        Args:
            data: EntityData from env.scene["robot"].data.
            joint_names: List of joint name strings.
            env_idx: Environment index to log. Defaults to self._env_idx.
        """
        idx = env_idx if env_idx is not None else self._env_idx

        # Joint positions and velocities
        joint_pos = data.joint_pos[idx].detach().cpu().numpy()
        joint_vel = data.joint_vel[idx].detach().cpu().numpy()
        for i, name in enumerate(joint_names):
            self.log_scalar(f"entity/joints/position/{name}", float(joint_pos[i]))
            self.log_scalar(f"entity/joints/velocity/{name}", float(joint_vel[i]))

        # Projected gravity
        grav = data.projected_gravity_b[idx].detach().cpu().numpy()
        for i, axis in enumerate(("x", "y", "z")):
            self.log_scalar(f"entity/gravity/{axis}", float(grav[i]))

        # Linear velocity
        lin_vel = data.root_link_lin_vel_b[idx].detach().cpu().numpy()
        for i, axis in enumerate(("x", "y", "z")):
            self.log_scalar(f"entity/velocity/linear/{axis}", float(lin_vel[i]))

        # Angular velocity
        ang_vel = data.root_link_ang_vel_b[idx].detach().cpu().numpy()
        for i, axis in enumerate(("x", "y", "z")):
            self.log_scalar(f"entity/velocity/angular/{axis}", float(ang_vel[i]))

        # Foot data
        foot_contact = data.foot_contact[idx].detach().cpu().numpy()
        foot_height = data.foot_height[idx].detach().cpu().numpy()
        foot_forces = data.foot_contact_forces[idx].detach().cpu().numpy()
        foot_air = data.foot_air_time[idx].detach().cpu().numpy()
        for i, foot in enumerate(_FOOT_NAMES):
            self.log_scalar(f"entity/feet/contact/{foot}", float(foot_contact[i]))
            self.log_scalar(f"entity/feet/height/{foot}", float(foot_height[i]))
            self.log_scalar(f"entity/feet/force/{foot}", float(foot_forces[i]))
            self.log_scalar(f"entity/feet/air_time/{foot}", float(foot_air[i]))

        # Velocity command
        cmd = data.vel_command[idx].detach().cpu().numpy()
        for i, label in enumerate(("vx", "vy", "wz")):
            self.log_scalar(f"entity/command/{label}", float(cmd[i]))

    def log_rewards(
        self,
        reward_manager: RewardManager,
        env_idx: int | None = None,
    ) -> None:
        """Log reward terms from the reward manager.

        Args:
            reward_manager: The env's reward manager.
            env_idx: Environment index. Defaults to self._env_idx.
        """
        idx = env_idx if env_idx is not None else self._env_idx

        # Total reward
        total = reward_manager._reward_buf[idx].detach().cpu().item()
        self.log_scalar("rewards/total", float(total))

        # Per-term rewards
        for term_idx, name in enumerate(reward_manager._term_names):
            val = reward_manager._step_reward[idx, term_idx].detach().cpu().item()
            self.log_scalar(f"rewards/{name}", float(val))

    def log_actions(
        self,
        action_manager: ActionManager,
        joint_names: list[str],
        env_idx: int | None = None,
    ) -> None:
        """Log raw and processed actions per joint.

        Args:
            action_manager: The env's action manager.
            joint_names: List of joint name strings.
            env_idx: Environment index. Defaults to self._env_idx.
        """
        idx = env_idx if env_idx is not None else self._env_idx

        # Raw actions
        raw = action_manager.action[idx].detach().cpu().numpy()
        for i, name in enumerate(joint_names):
            if i < len(raw):
                self.log_scalar(f"actions/raw/{name}", float(raw[i]))

        # Processed actions
        processed = action_manager.processed_action[idx].detach().cpu().numpy()
        for i, name in enumerate(joint_names):
            if i < len(processed):
                self.log_scalar(f"actions/processed/{name}", float(processed[i]))

    def log_terminations(
        self,
        termination_manager: TerminationManager,
        env_idx: int | None = None,
    ) -> None:
        """Log termination flags per term.

        Args:
            termination_manager: The env's termination manager.
            env_idx: Environment index. Defaults to self._env_idx.
        """
        idx = env_idx if env_idx is not None else self._env_idx

        for name in termination_manager._term_names:
            val = termination_manager._term_dones[name][idx].float().cpu().item()
            self.log_scalar(f"termination/{name}", float(val))

    def log_rl_step(self, env: ManagerBasedRlEnv, step: int, env_idx: int | None = None) -> None:
        """Log a full RL step: entity data, rewards, actions, terminations.

        Args:
            env: The ManagerBasedRlEnv instance.
            step: Current step counter.
            env_idx: Environment index. Defaults to self._env_idx.
        """
        idx = env_idx if env_idx is not None else self._env_idx
        joint_names = env.scene["robot"].joint_names

        self.set_step(step)
        self.log_entity_data(env.scene["robot"].data, joint_names, idx)
        self.log_rewards(env.reward_manager, idx)
        self.log_actions(env.action_manager, joint_names, idx)
        self.log_terminations(env.termination_manager, idx)

    def on_rl_step(self, env: ManagerBasedRlEnv, step: int) -> None:
        """Called from env.step() — respects log_interval.

        Args:
            env: The ManagerBasedRlEnv instance.
            step: Current step counter.
        """
        self._call_count += 1
        if self._call_count % self._log_interval != 0:
            return
        self.log_rl_step(env, step)

    # -- IL convenience methods --

    def log_observation(
        self,
        obs: dict | np.ndarray,
        names: list[str] | None = None,
    ) -> None:
        """Log observation data (dict or flat array).

        For dicts: each key is logged. Arrays with CHW shape are logged as images.
        For flat arrays: each index is logged as a scalar.

        Args:
            obs: Observation dict or numpy array.
            names: Optional names for array indices.
        """
        if isinstance(obs, dict):
            for key, val in obs.items():
                val = np.asarray(val)
                if val.ndim == 3 and val.shape[0] in (1, 3, 4) and val.shape[2] not in (1, 3, 4):
                    # CHW image
                    self.log_image(f"observation.image.{key}", val)
                elif val.ndim == 3 and val.shape[2] in (1, 3, 4):
                    # HWC image
                    self.log_image(f"observation.image.{key}", val)
                elif val.ndim <= 1:
                    flat = val.flatten()
                    for i, v in enumerate(flat):
                        self.log_scalar(f"observation.state/{key}_{i}", float(v))
                else:
                    flat = val.flatten()
                    for i, v in enumerate(flat):
                        self.log_scalar(f"observation.state/{key}_{i}", float(v))
        else:
            arr = np.asarray(obs).flatten()
            for i, v in enumerate(arr):
                label = names[i] if names and i < len(names) else str(i)
                self.log_scalar(f"observation.state/{label}", float(v))

    def log_action(
        self,
        action: dict | np.ndarray,
        names: list[str] | None = None,
    ) -> None:
        """Log action data (dict or flat array).

        Args:
            action: Action dict or numpy array.
            names: Optional names for array indices.
        """
        if isinstance(action, dict):
            for key, val in action.items():
                flat = np.asarray(val).flatten()
                for i, v in enumerate(flat):
                    self.log_scalar(f"action/{key}_{i}", float(v))
        else:
            arr = np.asarray(action).flatten()
            for i, v in enumerate(arr):
                label = names[i] if names and i < len(names) else str(i)
                self.log_scalar(f"action/{label}", float(v))

    def log_il_step(
        self,
        obs: dict | np.ndarray,
        action: dict | np.ndarray,
        step: int,
        obs_names: list[str] | None = None,
        action_names: list[str] | None = None,
    ) -> None:
        """Log a full IL step: set_step + log_observation + log_action.

        Args:
            obs: Observation dict or array.
            action: Action dict or array.
            step: Current step counter.
            obs_names: Optional names for observation indices.
            action_names: Optional names for action indices.
        """
        self.set_step(step)
        self.log_observation(obs, obs_names)
        self.log_action(action, action_names)
