#!/usr/bin/env python3
"""
Run a trained policy for inference/evaluation with optional keyboard control.

RL usage:
    python -m luckylab.scripts.play go2_velocity_flat --algorithm sac --checkpoint runs/go2_velocity_sac/checkpoints/best_agent.pt
    python -m luckylab.scripts.play go2_velocity_flat --algorithm ppo --checkpoint runs/go2_velocity_ppo/checkpoints/best_agent.pt --keyboard

IL usage:
    python -m luckylab.scripts.play so100_pickplace --policy act --checkpoint runs/luckylab_il/final

Keyboard controls (--keyboard mode, RL only):
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
class PlayRlConfig:
    """RL inference/evaluation configuration."""

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
    rerun: bool = False
    """Enable rerun live visualization."""
    rerun_save_path: str | None = None
    """Save rerun recording to .rrd file instead of spawning viewer."""


@dataclass(frozen=True)
class PlayIlConfig:
    """IL inference/evaluation configuration."""

    checkpoint: str
    """Path to the checkpoint directory (contains config.json + model.safetensors)."""
    policy: str = "act"
    """IL policy type (act, diffusion)."""
    device: str = "cpu"
    """Device to run inference on."""
    episodes: int = 10
    """Number of evaluation episodes."""
    rerun: bool = False
    """Enable rerun live visualization."""
    rerun_save_path: str | None = None
    """Save rerun recording to .rrd file instead of spawning viewer."""


def run_play_rl(task: str, cfg: PlayRlConfig) -> int:
    """Run RL evaluation with the given configuration."""
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

    if cfg.rerun:
        from luckylab.utils.rerun_logger import RerunLogger
        env.rerun_logger = RerunLogger(
            app_id="luckylab/play_rl",
            save_path=cfg.rerun_save_path,
            log_interval=1,
        )

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


def run_play_il(task: str, cfg: PlayIlConfig) -> int:
    """Run IL evaluation with the given configuration."""
    import torch

    from luckylab.il import IlRunnerCfg, load_policy
    from luckylab.il.lerobot.wrapper import make_lerobot_env
    from luckylab.tasks import load_il_cfg

    print_info(f"Loading task: {task} (IL mode)")

    # Get IL config for eval connection info
    il_cfg = load_il_cfg(task, cfg.policy)
    if il_cfg is None:
        il_cfg = IlRunnerCfg(policy=cfg.policy)
        print_info(f"Using default IL configuration for {cfg.policy.upper()}")

    # Load policy + preprocessor/postprocessor
    print_info(f"Loading {cfg.policy.upper()} checkpoint: {cfg.checkpoint}")
    try:
        policy, preprocessor, postprocessor = load_policy(
            checkpoint_path=cfg.checkpoint,
            il_cfg=il_cfg,
            device=cfg.device,
        )
    except FileNotFoundError:
        print_info(f"Checkpoint not found: {cfg.checkpoint}", color="red")
        return 1

    # Extract obs/action dims and camera names from the policy's input/output features.
    obs_dim = 0
    action_dim = 0
    camera_names = []
    camera_shape = (3, 256, 256)
    if hasattr(policy.config, "input_features"):
        for key, feat in policy.config.input_features.items():
            if key == "observation.state":
                obs_dim = feat.shape[0]
            elif key.startswith("observation.images."):
                cam_name = key.split("observation.images.", 1)[1]
                camera_names.append(cam_name)
                camera_shape = tuple(feat.shape)
                print_info(f"  Camera: {cam_name} ({camera_shape})")
    if hasattr(policy.config, "output_features"):
        for key, feat in policy.config.output_features.items():
            if key == "action":
                action_dim = feat.shape[0]

    # Connect to LuckyEngine for eval
    env = make_lerobot_env(
        il_cfg,
        obs_dim=obs_dim,
        action_dim=action_dim,
        camera_names=camera_names if camera_names else None,
        camera_width=camera_shape[-1],
        camera_height=camera_shape[-2],
    )

    # Run evaluation
    rr_log = None
    if cfg.rerun:
        from luckylab.utils.rerun_logger import RerunLogger
        rr_log = RerunLogger(
            app_id="luckylab/play_il",
            save_path=cfg.rerun_save_path,
            log_interval=1,
        )

    print_info(f"Running IL evaluation for {cfg.episodes} episodes...")
    episode_lengths = []

    try:
        for ep in range(cfg.episodes):
            obs, _ = env.reset()
            policy.reset()
            steps = 0

            for _ in range(10_000):  # Max steps per episode
                # Convert obs to tensors and normalize via preprocessor
                batch = {}
                for key, val in obs.items():
                    batch[key] = torch.from_numpy(val).unsqueeze(0).to(cfg.device)
                batch = preprocessor(batch)

                action = policy.select_action(batch)

                # Denormalize action and move to CPU via postprocessor
                action = postprocessor(action)
                action_np = action.squeeze(0).cpu().numpy()

                obs, _, terminated, truncated, _ = env.step(action_np)
                steps += 1

                if rr_log is not None:
                    rr_log.log_il_step(obs, action_np, step=steps)

                if terminated or truncated:
                    break

            episode_lengths.append(steps)
            print_info(f"Episode {ep + 1}: length={steps}")

    except KeyboardInterrupt:
        print_info("Evaluation interrupted by user", color="yellow")
    finally:
        if rr_log is not None:
            rr_log.close()
        env.close()

    if episode_lengths:
        print()
        print_header("IL Evaluation Results")
        print_info(f"  Policy:        {cfg.policy.upper()}")
        print_info(f"  Episodes:      {len(episode_lengths)}")
        print_info(f"  Mean Length:   {np.mean(episode_lengths):.1f}")
        print_info(f"  Min Length:    {np.min(episode_lengths)}")
        print_info(f"  Max Length:    {np.max(episode_lengths)}")

    print_info("IL evaluation complete!")
    return 0


def main() -> int:
    import luckylab.tasks  # noqa: F401
    from luckylab.tasks import list_il_policies, list_rl_policies, list_tasks

    all_tasks = list_tasks()
    if not all_tasks:
        print_info("No tasks registered!", color="red")
        return 1

    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )

    has_policy_arg = any(arg.startswith("--policy") for arg in remaining_args)
    has_algorithm_arg = any(arg.startswith("--algorithm") for arg in remaining_args)
    has_rl = bool(list_rl_policies(chosen_task))
    has_il = bool(list_il_policies(chosen_task))

    if has_policy_arg and not has_algorithm_arg:
        mode = "il"
    elif has_algorithm_arg and not has_policy_arg:
        mode = "rl"
    elif has_il and not has_rl:
        mode = "il"
    elif has_rl and not has_il:
        mode = "rl"
    else:
        print_info(
            f"Cannot determine mode for task '{chosen_task}'. "
            "Use --algorithm for RL or --policy for IL.",
            color="red",
        )
        return 1

    if mode == "il":
        cfg = tyro.cli(
            PlayIlConfig,
            args=remaining_args,
            prog=sys.argv[0] + f" {chosen_task}",
        )
        return run_play_il(chosen_task, cfg)
    else:
        cfg = tyro.cli(
            PlayRlConfig,
            args=remaining_args,
            prog=sys.argv[0] + f" {chosen_task}",
        )
        return run_play_rl(chosen_task, cfg)


if __name__ == "__main__":
    sys.exit(main())
