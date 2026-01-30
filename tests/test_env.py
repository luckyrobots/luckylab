"""Tests for luckylab environments."""

import luckylab  # noqa: F401 - registers environments
from luckylab.envs import ManagerBasedRlEnvCfg
from luckylab.managers import CurriculumTermCfg, RewardTermCfg, TerminationTermCfg
from luckylab.tasks.velocity import create_velocity_env_cfg
from luckylab.tasks.velocity.mdp import (
    fall_termination,
    terrain_curriculum,
    track_linear_velocity,
)


def test_manager_based_rl_env_cfg_defaults():
    """Test that ManagerBasedRlEnvCfg has expected default values."""
    # decimation is required (mjlab pattern)
    cfg = ManagerBasedRlEnvCfg(decimation=4)
    assert cfg.decimation == 4
    assert cfg.scene == "velocity"
    assert cfg.task == "locomotion"
    assert cfg.robot == "unitreego1"
    assert cfg.skip_launch is True
    # max_episode_length is derived from episode_length_s / step_dt
    assert cfg.episode_length_s == 20.0
    assert cfg.max_episode_length == 250  # 20.0 / (0.02 * 4)
    # Default has empty rewards/terminations (config-driven)
    assert isinstance(cfg.rewards, dict)
    assert isinstance(cfg.terminations, dict)
    assert len(cfg.rewards) == 0
    assert len(cfg.terminations) == 0
    # commands and curriculum default to None (mjlab pattern)
    assert cfg.commands is None
    assert cfg.curriculum is None


def test_create_velocity_env_cfg():
    """Test the factory function creates a complete config."""
    cfg = create_velocity_env_cfg(robot="unitreego1")
    assert cfg.robot == "unitreego1"
    assert len(cfg.rewards) > 0
    assert len(cfg.terminations) > 0
    assert len(cfg.curriculum) > 0
    assert "track_linear_velocity" in cfg.rewards
    assert "fell_over" in cfg.terminations
    assert "terrain_difficulty" in cfg.curriculum
    assert "command_velocity" in cfg.curriculum


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
        func=fall_termination,
        params={"z_threshold": -2.0},
    )
    assert term.func is fall_termination
    assert term.params == {"z_threshold": -2.0}


def test_curriculum_term_cfg():
    """Test CurriculumTermCfg configuration."""
    term = CurriculumTermCfg(
        func=terrain_curriculum,
        params={
            "stages": [
                {"step": 0, "difficulty": 0.0},
                {"step": 100, "difficulty": 1.0},
            ]
        },
    )
    assert term.func is terrain_curriculum
    assert "stages" in term.params
    assert len(term.params["stages"]) == 2


def test_env_registration():
    """Test that the environment is registered with gymnasium."""
    import gymnasium as gym

    # Check that the environment is registered
    env_spec = gym.spec("luckylab/Go1-Velocity-v0")
    assert env_spec is not None
    assert env_spec.id == "luckylab/Go1-Velocity-v0"
    # max_episode_steps = episode_length_s / (sim_dt * decimation) = 20.0 / (0.02 * 4) = 250
    assert env_spec.max_episode_steps == 250
