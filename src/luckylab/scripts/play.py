#!/usr/bin/env python3
"""
Run a trained policy for inference/evaluation with optional keyboard control.

Usage:
    python -m luckylab.scripts.play go2_velocity_flat --algorithm sac --checkpoint runs/go2_velocity_sac/checkpoints/best_agent.pt
    python -m luckylab.scripts.play go2_velocity_flat --algorithm ppo --checkpoint runs/go2_velocity_ppo/checkpoints/best_agent.pt --keyboard

Keyboard controls (--keyboard mode):
    W/S: forward/backward    A/D: strafe left/right    Q/E: turn left/right
    Space: zero all commands  Esc: quit
"""

import copy
import sys
from dataclasses import dataclass

import numpy as np
import tyro

from luckylab.utils.logging import print_header, print_info


@dataclass(frozen=True)
class PlayConfig:
    """Inference/evaluation configuration."""

    checkpoint: str
    """Path to the checkpoint file."""
    algorithm: str
    """RL algorithm used for training (ppo, sac, td3, ddpg)."""
    device: str = "cpu"
    """Device to run inference on."""
    episodes: int = 10
    """Number of evaluation episodes."""
    keyboard: bool = False
    """Enable keyboard velocity command control."""


def run_play(task: str, cfg: PlayConfig) -> int:
    """Run evaluation with the given configuration."""
    from luckylab.rl import RlRunnerCfg, load_agent
    from luckylab.tasks import load_env_cfg, load_rl_cfg

    print_info(f"Loading task: {task}")
    try:
        env_cfg = load_env_cfg(task)
    except KeyError as e:
        print_info(str(e), color="red")
        return 1

    env_cfg.simulation_mode = "realtime"

    rl_cfg = load_rl_cfg(task, cfg.algorithm)
    if rl_cfg is None:
        rl_cfg = RlRunnerCfg(algorithm=cfg.algorithm)
        print_info(f"Using default RL configuration for {cfg.algorithm.upper()}")
    else:
        rl_cfg = copy.deepcopy(rl_cfg)
        print_info(f"Using {cfg.algorithm.upper()} configuration")

    print_info(f"Loading checkpoint: {cfg.checkpoint}")
    try:
        agent, wrapped_env = load_agent(
            checkpoint_path=cfg.checkpoint,
            env_cfg=env_cfg,
            rl_cfg=rl_cfg,
            device=cfg.device,
        )
    except FileNotFoundError:
        print_info(f"Checkpoint not found: {cfg.checkpoint}", color="red")
        return 1

    agent.set_running_mode("eval")
    agent.set_mode("eval")

    env = wrapped_env.unwrapped
    robot_data = env.scene["robot"].data

    kb = None
    if cfg.keyboard:
        from luckylab.utils.keyboard import KeyboardController

        kb = KeyboardController()
        kb.start()
        print_info("Keyboard control enabled:")
        print_info("  W/S: forward/backward  A/D: strafe left/right  Q/E: turn left/right")
        print_info("  Space: zero commands   Esc: quit")

    if kb:
        # Keyboard mode: run continuously, auto-reset on termination, Esc to quit
        print_info("Running continuously (Esc to quit)...")
        obs, _ = wrapped_env.reset()
        total_steps = 0
        ep_reward = 0.0
        ep_steps = 0

        try:
            while not kb.should_quit:
                vx, vy, wz = kb.get_command()
                # Override engine-sampled command with keyboard input
                robot_data._vel_command[:, 0] = vx
                robot_data._vel_command[:, 1] = vy
                robot_data._vel_command[:, 2] = wz

                actions, _, _ = agent.act(obs, timestep=0, timesteps=0)
                obs, reward, terminated, truncated, _ = wrapped_env.step(actions)

                ep_reward += reward.item()
                ep_steps += 1
                total_steps += 1

                sys.stdout.write(
                    f"\r  cmd: vx={vx:+.1f} vy={vy:+.1f} wz={wz:+.2f}  "
                    f"step={ep_steps}  reward={ep_reward:.1f}   "
                )
                sys.stdout.flush()

                if terminated.item() or truncated.item():
                    sys.stdout.write(f"\n  reset (ep reward={ep_reward:.1f}, len={ep_steps})\n")
                    obs, _ = wrapped_env.reset()
                    ep_reward = 0.0
                    ep_steps = 0

        except KeyboardInterrupt:
            print_info("\nInterrupted by user", color="yellow")
        finally:
            sys.stdout.write("\n")
            kb.stop()
            wrapped_env.close()

        print_info(f"Total steps: {total_steps}")
        print_info("Done!")
        return 0

    # Non-keyboard mode: fixed number of episodes
    print_info(f"Running evaluation for {cfg.episodes} episodes...")
    episode_rewards = []
    episode_lengths = []

    try:
        for ep in range(cfg.episodes):
            obs, _ = wrapped_env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                actions, _, _ = agent.act(obs, timestep=0, timesteps=0)
                obs, reward, terminated, truncated, _ = wrapped_env.step(actions)

                total_reward += reward.item()
                steps += 1
                done = terminated.item() or truncated.item()

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            print_info(f"Episode {ep + 1}: reward={total_reward:.2f}, length={steps}")

    except KeyboardInterrupt:
        print_info("Evaluation interrupted by user", color="yellow")
    finally:
        wrapped_env.close()

    if episode_rewards:
        print()
        print_header("Evaluation Results")
        print_info(f"  Algorithm:     {cfg.algorithm.upper()}")
        print_info(f"  Episodes:      {len(episode_rewards)}")
        print_info(f"  Mean Reward:   {np.mean(episode_rewards):.2f}")
        print_info(f"  Std Reward:    {np.std(episode_rewards):.2f}")
        print_info(f"  Min Reward:    {np.min(episode_rewards):.2f}")
        print_info(f"  Max Reward:    {np.max(episode_rewards):.2f}")
        print_info(f"  Mean Length:   {np.mean(episode_lengths):.1f}")

    print_info("Evaluation complete!")
    return 0


def main() -> int:
    import luckylab.tasks  # noqa: F401
    from luckylab.tasks import list_tasks

    all_tasks = list_tasks()
    if not all_tasks:
        print_info("No tasks registered!", color="red")
        return 1

    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )

    cfg = tyro.cli(
        PlayConfig,
        args=remaining_args,
        prog=sys.argv[0] + f" {chosen_task}",
    )

    return run_play(chosen_task, cfg)


if __name__ == "__main__":
    sys.exit(main())
