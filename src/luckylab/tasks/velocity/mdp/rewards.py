from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from luckylab.entity import Entity
from luckylab.managers.manager_term_config import RewardTermCfg
from luckylab.utils.math import quat_apply_inverse
from luckylab.managers.scene_entity_config import SceneEntityCfg
from luckylab.utils.string import resolve_matching_names_values

if TYPE_CHECKING:
    from luckylab.envs.manager_based_rl_env import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def move_in_command_direction(
    env: ManagerBasedRlEnv,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for moving in the commanded direction.

    Projects actual XY velocity onto the commanded XY direction and clamps to [0, 1].
    Gated by command magnitude to avoid free reward for near-zero commands.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = asset.data.vel_command
    cmd_xy = command[:, :2]  # [B, 2]
    cmd_norm = torch.norm(cmd_xy, dim=1, keepdim=True)  # [B, 1]
    active = (cmd_norm.squeeze(1) > command_threshold).float()  # [B]
    cmd_norm = cmd_norm.clamp(min=1e-6)
    cmd_dir = cmd_xy / cmd_norm  # [B, 2] unit direction
    actual_xy = asset.data.root_link_lin_vel_b[:, :2]  # [B, 2]
    projection = torch.sum(actual_xy * cmd_dir, dim=1) / cmd_norm.squeeze(1)  # [B]
    return torch.clamp(projection, 0.0, 1.0) * active


def trot_contact(
    env: ManagerBasedRlEnv,
    speed_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward diagonal contact pattern (trot gait).

    Rewards when diagonal leg pairs (FL+RR, FR+RL) are in opposite phases —
    one pair in contact while the other swings. Speed-gated to prevent
    earning reward while stationary (e.g. front-feet endo).
    """
    asset: Entity = env.scene[asset_cfg.name]
    contact = asset.data.foot_contact  # [B, 4] order: FL, FR, RL, RR
    # Diagonal pairs: FL(0)+RR(3) and FR(1)+RL(2)
    # Perfect trot: one pair in contact, other in air
    # XOR-like: reward when contact states differ within a diagonal pair
    pair1_match = torch.abs(contact[:, 0] - contact[:, 3])  # FL vs RR should be same
    pair2_match = torch.abs(contact[:, 1] - contact[:, 2])  # FR vs RL should be same
    # Pairs should be in sync (both match = 0), and out of phase with each other
    pair1_sync = 1.0 - pair1_match  # 1 when FL and RR have same contact state
    pair2_sync = 1.0 - pair2_match  # 1 when FR and RL have same contact state
    # Pairs should be in opposite phase
    pair1_contact = (contact[:, 0] + contact[:, 3]) / 2.0  # avg contact of pair 1
    pair2_contact = (contact[:, 1] + contact[:, 2]) / 2.0  # avg contact of pair 2
    opposite_phase = torch.abs(pair1_contact - pair2_contact)  # 1 when perfectly opposite
    reward = pair1_sync * pair2_sync * opposite_phase
    # Speed gate: only reward when actually moving
    actual_vel = asset.data.root_link_lin_vel_b[:, :2]
    actual_speed = torch.norm(actual_vel, dim=1)
    speed_gate = (actual_speed > speed_threshold).float()
    return reward * speed_gate


def excessive_tilt(
    env: ManagerBasedRlEnv,
    safe_angle: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize body tilt beyond a safe angle (in degrees).

    Returns 1.0 when tilt exceeds safe_angle, 0.0 otherwise.
    Acts as an early warning before the fell_over termination.
    """
    asset: Entity = env.scene[asset_cfg.name]
    if asset_cfg.body_ids:
        body_quat_w = asset.data.root_link_quat_w
        gravity_w = asset.data.gravity_vec_w
        projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
    else:
        projected_gravity_b = asset.data.projected_gravity_b
    # gravity_z component: 1.0 when upright, 0.0 when horizontal
    # tilt angle = acos(|gravity_z|)
    gravity_z = torch.abs(projected_gravity_b[:, 2])
    safe_cos = torch.cos(torch.tensor(safe_angle * torch.pi / 180.0))
    return (gravity_z < safe_cos).float()


def track_linear_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Baselined exp kernel for linear velocity tracking.

    Returns exp(-error²/std²) - exp(-||cmd||²/std²). The baseline subtraction
    ensures standing still earns exactly 0 regardless of command magnitude,
    eliminating free credit that causes the standing-still plateau.
    Can go negative (moving wrong direction is worse than standing).
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = asset.data.vel_command
    actual = asset.data.root_link_lin_vel_b
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    error_sq = xy_error + z_error
    tracking = torch.exp(-error_sq / std**2)
    cmd_sq = torch.sum(torch.square(command[:, :2]), dim=1)
    baseline = torch.exp(-cmd_sq / std**2)
    return tracking - baseline


def track_angular_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Baselined exp kernel for angular velocity tracking (yaw only).

    Returns exp(-error²/std²) - exp(-cmd_z²/std²). Baseline subtraction
    ensures standing still earns 0. Yaw-only means no roll/pitch punishment
    bombs during stumbles (handled by upright reward instead).
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = asset.data.vel_command
    actual = asset.data.root_link_ang_vel_b
    error_sq = torch.square(command[:, 2] - actual[:, 2])
    tracking = torch.exp(-error_sq / std**2)
    baseline = torch.exp(-torch.square(command[:, 2]) / std**2)
    return tracking - baseline


def flat_orientation(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward flat base orientation (robot being upright).

    If asset_cfg has body_ids specified, computes the projected gravity
    for that specific body. Otherwise, uses the root link projected gravity.
    """
    asset: Entity = env.scene[asset_cfg.name]

    # If body_ids are specified, compute projected gravity for that body.
    if asset_cfg.body_ids:
        body_quat_w = asset.data.root_link_quat_w  # [B, 4]
        gravity_w = asset.data.gravity_vec_w  # [B, 3]
        projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
        xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    else:
        # Use root link projected gravity.
        xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

    return torch.exp(-xy_squared / std**2)


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Penalize self-collisions.

    Not yet available in LuckyEngine. Returns zeros so it can be included
    in configs at weight 0.0 without crashing.
    """
    return torch.zeros(env.num_envs, device=env.device)


def body_angular_velocity_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize excessive body angular velocities in world frame."""
    asset: Entity = env.scene[asset_cfg.name]
    ang_vel = asset.data.root_link_ang_vel_w
    ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
    return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Penalize whole-body angular momentum to encourage natural arm swing.

    Not yet available in LuckyEngine. Returns zeros so it can be included
    in configs at weight 0.0 without crashing.
    """
    return torch.zeros(env.num_envs, device=env.device)


def feet_air_time(
    env: ManagerBasedRlEnv,
    threshold_min: float = 0.05,
    threshold_max: float = 0.5,
    command_threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward feet air time. Gated by command magnitude to avoid rewarding while stationary."""
    asset: Entity = env.scene[asset_cfg.name]
    current_air_time = asset.data.foot_air_time
    assert current_air_time is not None, "foot_air_time is None - ensure LuckyEngine provides this data"
    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)
    in_air = current_air_time > 0
    num_in_air = torch.sum(in_air.float())
    mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
        num_in_air, min=1
    )
    env.extras.setdefault("episode", {})["Metrics/air_time_mean"] = mean_air_time
    command = asset.data.vel_command
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    scale = (total_command > command_threshold).float()
    reward *= scale
    return reward


def feet_clearance(
    env: ManagerBasedRlEnv,
    target_height: float,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize deviation from target clearance height, weighted by foot velocity.

    Matches mjlab: penalizes foot height deviation from target more when the foot
    is moving fast horizontally (during swing phase). Gated by command magnitude.
    """
    asset: Entity = env.scene[asset_cfg.name]
    foot_z = asset.data.foot_height  # [B, 4]
    foot_vel = asset.data.foot_velocity  # [B, 4, 3]
    foot_vel_xy = foot_vel[:, :, :2]  # [B, 4, 2]
    vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, 4]
    delta = torch.abs(foot_z - target_height)  # [B, 4]
    in_air = (asset.data.foot_contact < 0.5).float()  # [B, 4]
    num_in_air = torch.sum(in_air)
    mean_clearance = torch.sum(foot_z * in_air) / torch.clamp(num_in_air, min=1)
    env.extras.setdefault("episode", {})["Metrics/foot_clearance_mean"] = mean_clearance
    cost = torch.sum(delta * vel_norm, dim=1)  # [B]
    command = asset.data.vel_command
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    cost = cost * active
    return cost


class feet_swing_height:
    """Penalize deviation from target swing height, evaluated at landing.

    Adapted from mjlab: uses foot_height and foot_contact instead of
    ContactSensor (which is not available from LuckyEngine).
    Tracks peak foot height during swing phase and evaluates error at first contact.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.peak_heights = torch.zeros(env.num_envs, 4, device=env.device)
        self.prev_contact = torch.zeros(env.num_envs, 4, device=env.device)
        self.step_dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        target_height: float,
        command_threshold: float = 0.05,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        command = asset.data.vel_command
        foot_heights = asset.data.foot_height  # [B, 4]
        contact = asset.data.foot_contact  # [B, 4]
        in_air = (contact < 0.5).float()
        self.peak_heights = torch.where(
            in_air.bool(),
            torch.maximum(self.peak_heights, foot_heights),
            self.peak_heights,
        )
        first_contact = ((self.prev_contact < 0.5) & (contact > 0.5)).float()
        self.prev_contact = contact.clone()

        # Compute cost at landing
        linear_norm = torch.norm(command[:, :2], dim=1)
        angular_norm = torch.abs(command[:, 2])
        total_command = linear_norm + angular_norm
        active = (total_command > command_threshold).float()
        error = self.peak_heights / target_height - 1.0
        cost = torch.sum(torch.square(error) * first_contact, dim=1) * active

        # Log mean peak height at landing
        num_landings = torch.sum(first_contact)
        peak_at_landing = self.peak_heights * first_contact
        mean_peak_height = torch.sum(peak_at_landing) / torch.clamp(num_landings, min=1)
        env.extras.setdefault("episode", {})["Metrics/peak_height_mean"] = mean_peak_height

        self.peak_heights = torch.where(
            first_contact.bool(),
            torch.zeros_like(self.peak_heights),
            self.peak_heights,
        )
        return cost


def feet_slip(
    env: ManagerBasedRlEnv,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize foot sliding (XY velocity while in contact)."""
    asset: Entity = env.scene[asset_cfg.name]
    command = asset.data.vel_command
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    in_contact = (asset.data.foot_contact > 0.5).float()  # [B, 4]
    foot_vel_xy = asset.data.foot_velocity[:, :, :2]  # [B, 4, 2]
    vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, 4]
    vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, 4]
    cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
    num_in_contact = torch.sum(in_contact)
    mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(num_in_contact, min=1)
    env.extras.setdefault("episode", {})["Metrics/slip_velocity_mean"] = mean_slip_vel
    return cost


def soft_landing(
    env: ManagerBasedRlEnv,
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize high impact forces at landing to encourage soft footfalls.

    Adapted from mjlab: uses foot_contact_forces and foot_contact instead of
    ContactSensor (which is not available from LuckyEngine). Gated by command magnitude.
    """
    asset: Entity = env.scene[asset_cfg.name]
    force_magnitude = asset.data.foot_contact_forces  # [B, 4]
    contact = asset.data.foot_contact  # [B, 4]

    # Detect first contact using stored previous state
    if not hasattr(soft_landing, "_prev_contact"):
        soft_landing._prev_contact = torch.zeros_like(contact)

    first_contact = ((soft_landing._prev_contact < 0.5) & (contact > 0.5)).float()
    soft_landing._prev_contact = contact.clone()

    landing_impact = force_magnitude * first_contact  # [B, 4]
    cost = torch.sum(landing_impact, dim=1)  # [B]
    num_landings = torch.sum(first_contact)
    mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
    env.extras.setdefault("episode", {})["Metrics/landing_force_mean"] = mean_landing_force

    command = asset.data.vel_command
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    cost = cost * active
    return cost


class variable_posture:
    """Penalize deviation from default pose, with tighter constraints when standing."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        default_joint_pos = asset.data.default_joint_pos
        assert default_joint_pos is not None
        self.default_joint_pos = default_joint_pos

        joint_names = asset.data.joint_names

        _, _, std_standing = resolve_matching_names_values(
            data=cfg.params["std_standing"],
            list_of_strings=joint_names,
        )
        self.std_standing = torch.tensor(
            std_standing, device=env.device, dtype=torch.float32
        )

        _, _, std_walking = resolve_matching_names_values(
            data=cfg.params["std_walking"],
            list_of_strings=joint_names,
        )
        self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

        _, _, std_running = resolve_matching_names_values(
            data=cfg.params["std_running"],
            list_of_strings=joint_names,
        )
        self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std_standing,
        std_walking,
        std_running,
        walking_threshold: float = 0.5,
        running_threshold: float = 1.5,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        del std_standing, std_walking, std_running  # Unused

        asset: Entity = env.scene[asset_cfg.name]
        command = asset.data.vel_command

        linear_speed = torch.norm(command[:, :2], dim=1)
        angular_speed = torch.abs(command[:, 2])
        total_speed = linear_speed + angular_speed

        standing_mask = (total_speed < walking_threshold).float()
        walking_mask = (
            (total_speed >= walking_threshold) & (total_speed < running_threshold)
        ).float()
        running_mask = (total_speed >= running_threshold).float()

        std = (
            self.std_standing * standing_mask.unsqueeze(1)
            + self.std_walking * walking_mask.unsqueeze(1)
            + self.std_running * running_mask.unsqueeze(1)
        )

        current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
        error_squared = torch.square(current_joint_pos - desired_joint_pos)

        return torch.exp(-torch.mean(error_squared / (std**2), dim=1))
