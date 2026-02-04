#!/usr/bin/env python3
"""List all registered LuckyLab environments and tasks."""

import luckylab  # noqa: F401 - registers environments
from luckylab.tasks import list_tasks, load_env_cfg


def main():
    print("=" * 60)
    print("LuckyLab Registered Tasks")
    print("=" * 60)

    tasks = list_tasks()

    if not tasks:
        print("No tasks registered.")
        return

    for task_id in tasks:
        cfg = load_env_cfg(task_id)
        print(f"\n  {task_id}")
        print(f"    Robot: {cfg.robot}")
        print(f"    Scene: {cfg.scene}")
        print(f"    Task:  {cfg.task}")
        print(f"    Episode Length: {cfg.episode_length_s}s")

    print("\n" + "=" * 60)
    print("Gymnasium Environments")
    print("=" * 60)
    print("\n  luckylab/UnitreeGo1-Locomotion-v0")
    print("    Default environment for Unitree Go1 velocity tracking")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
