"""Tests for luckylab environments."""

from luckylab.envs import ManagerBasedRlEnvCfg
from luckylab.managers.manager_term_config import (
    CurriculumTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from luckylab.tasks.velocity.mdp import (
    bad_orientation,
    terrain_levels_vel,
    track_linear_velocity,
)


def test_manager_based_rl_env_cfg_defaults():
    """Test that ManagerBasedRlEnvCfg has expected default values."""
    cfg = ManagerBasedRlEnvCfg(decimation=4, observations={}, actions={})
    assert cfg.decimation == 4
    assert cfg.scene == "velocity"
    assert cfg.task == "locomotion"
    assert cfg.robot == "unitreego2"
    assert cfg.skip_launch is True
    # max_episode_length is derived from episode_length_s / step_dt
    assert cfg.episode_length_s == 20.0
    # Default has empty rewards/terminations (config-driven)
    assert isinstance(cfg.rewards, dict)
    assert isinstance(cfg.terminations, dict)
    assert len(cfg.rewards) == 0
    assert len(cfg.terminations) == 0
    # curriculum defaults to None, simulation_contract has defaults
    assert cfg.curriculum is None
    assert cfg.simulation_contract is not None
    assert cfg.simulation_contract.vel_command_x_range == (-1.0, 1.0)


def test_reward_term_cfg():
    """Test RewardTermCfg configuration."""
    term = RewardTermCfg(
        func=track_linear_velocity,
        weight=2.0,
        params={"std": 0.5},
    )
    assert term.func is track_linear_velocity
    assert term.weight == 2.0
    assert term.params == {"std": 0.5}


def test_termination_term_cfg():
    """Test TerminationTermCfg configuration."""
    term = TerminationTermCfg(
        func=bad_orientation,
        params={"limit_angle": 1.22},
    )
    assert term.func is bad_orientation
    assert term.params == {"limit_angle": 1.22}


def test_curriculum_term_cfg():
    """Test CurriculumTermCfg configuration."""
    term = CurriculumTermCfg(
        func=terrain_levels_vel,
        params={
            "command_name": "twist",
        },
    )
    assert term.func is terrain_levels_vel
    assert "command_name" in term.params
