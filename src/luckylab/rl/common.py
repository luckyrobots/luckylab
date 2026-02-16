"""Shared helpers used by both skrl and sb3 backends."""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

from prettytable import PrettyTable

from luckylab.utils.logging import print_info

if TYPE_CHECKING:
    from luckylab.envs import ManagerBasedRlEnv
    from luckylab.rl.config import RlRunnerCfg


def make_experiment_name(rl_cfg: RlRunnerCfg) -> str:
    """Strip any algorithm suffix and re-add the current one."""
    base = rl_cfg.experiment_name
    for suffix in ("_ppo", "_sac", "_td3", "_ddpg"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return f"{base}_{rl_cfg.algorithm}"


def print_config(
    env: ManagerBasedRlEnv,
    rl_cfg: RlRunnerCfg,
    experiment_name: str,
    device: str,
) -> None:
    """Print consolidated training summary after env is initialized."""
    env_cfg = env.cfg
    algo_cfg = getattr(rl_cfg, rl_cfg.algorithm)
    policy = rl_cfg.policy

    # --- Header ---
    print_info("")
    print_info(f"  {experiment_name}")
    print_info("")

    # --- Training + Environment (single table) ---
    t = PrettyTable(field_names=["Key", "Value"])
    t.header = False
    t.align = "l"
    t.title = "Configuration"

    t.add_row(["Algorithm", f"{rl_cfg.algorithm.upper()} ({rl_cfg.backend})"])
    t.add_row(["Device", device])
    t.add_row(["Timesteps", f"{rl_cfg.max_iterations:,}"])
    t.add_row(["Seed", rl_cfg.seed])
    t.add_row(["Sim Mode", env_cfg.simulation_mode])
    t.add_row(["Episode", f"{env_cfg.episode_length_s}s ({env.max_episode_length} steps @ {1/env.step_dt:.0f} Hz)"])
    if rl_cfg.wandb:
        t.add_row(["Wandb", rl_cfg.wandb_project])

    t.add_row(["", ""])  # separator

    # Network
    t.add_row(["Actor", f"{policy.actor_hidden_dims}  {policy.activation}"])
    t.add_row(["Critic", f"{policy.critic_hidden_dims}  {policy.activation}"])
    if policy.noise_type == "gsde":
        t.add_row(["Exploration", f"gSDE (std={policy.init_noise_std}, resample={policy.gsde_resample_interval})"])
    else:
        t.add_row(["Noise Std", policy.init_noise_std])
    if policy.use_delta_actions:
        t.add_row(["Delta Actions", f"scale={policy.delta_action_scale}"])

    t.add_row(["", ""])  # separator

    # Algorithm hyperparams (compact)
    for f in fields(algo_cfg):
        val = getattr(algo_cfg, f.name)
        label = f.name.replace("_", " ").title()
        if isinstance(val, int) and val >= 10_000:
            t.add_row([label, f"{val:,}"])
        else:
            t.add_row([label, val])

    print_info(t.get_string())
    print_info("")

    # --- Observations (compact) ---
    obs_mgr = env.observation_manager
    for group_name in obs_mgr.active_terms:
        term_names = obs_mgr.active_terms[group_name]
        term_dims = obs_mgr.group_obs_term_dim[group_name]
        total = sum(d[0] if isinstance(d, tuple) else d for d in term_dims)
        parts = [f"{name}({d[0] if isinstance(d, tuple) else d})" for name, d in zip(term_names, term_dims)]
        print_info(f"  Observations [{group_name}] ({total}):")
        line = "    "
        for part in parts:
            if len(line) + len(part) + 2 > 80:
                print_info(line)
                line = "    "
            line += part + "  "
        if line.strip():
            print_info(line)

    # --- Actions (compact) ---
    act_mgr = env.action_manager
    parts = [f"{name}({d})" for name, d in zip(act_mgr.active_terms, act_mgr.action_term_dim)]
    print_info(f"  Actions ({act_mgr.total_action_dim}): {', '.join(parts)}")
    print_info("")

    # --- Rewards (skip weight=0) ---
    rew_mgr = env.reward_manager
    active_rewards = [
        (name, cfg.weight)
        for name, cfg in zip(rew_mgr.active_terms, rew_mgr._term_cfgs)
        if cfg.weight != 0.0
    ]
    if active_rewards:
        parts = [f"{name}({w:g})" for name, w in active_rewards]
        print_info(f"  Rewards: {', '.join(parts)}")

    # --- Terminations ---
    if env.termination_manager.active_terms:
        print_info(f"  Terminations: {', '.join(env.termination_manager.active_terms)}")

    # --- Curriculum ---
    if env.curriculum_manager.active_terms:
        print_info(f"  Curriculum: {', '.join(env.curriculum_manager.active_terms)}")

    print_info("")


def wrap_env(env, rl_cfg: RlRunnerCfg, wrapper_cls):
    """Create a wrapper with algorithm-specific settings.

    Args:
        env: The ManagerBasedRlEnv instance.
        rl_cfg: RL runner configuration.
        wrapper_cls: SkrlWrapper or Sb3Wrapper class.
    """
    algo_cfg = getattr(rl_cfg, rl_cfg.algorithm)
    kwargs: dict = {
        "clip_actions": getattr(algo_cfg, "clip_actions", None),
    }
    if rl_cfg.policy.use_delta_actions:
        kwargs["use_delta_actions"] = True
        kwargs["delta_action_scale"] = rl_cfg.policy.delta_action_scale
    return wrapper_cls(env, **kwargs)
