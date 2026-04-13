#!/usr/bin/env python3
"""List all registered LuckyLab environments and tasks."""

import luckylab  # noqa: F401 - registers environments
from luckylab.tasks import list_il_policies, list_rl_policies, list_tasks, load_env_cfg, load_il_cfg


def main():
    print("=" * 60)
    print("LuckyLab Registered Tasks")
    print("=" * 60)

    tasks = list_tasks()

    if not tasks:
        print("No tasks registered.")
        return

    for task_id in tasks:
        print(f"\n  {task_id}")

        # RL tasks have an env config
        try:
            cfg = load_env_cfg(task_id)
            print(f"    Robot: {cfg.robot}")
            print(f"    Scene: {cfg.scene}")
            print(f"    Task:  {cfg.task}")
            print(f"    Episode Length: {cfg.episode_length_s}s")
        except ValueError:
            pass

        # IL-only tasks get their info from the IL config
        il_policies = list_il_policies(task_id)
        if il_policies:
            il_cfg = load_il_cfg(task_id, il_policies[0])
            if il_cfg:
                try:
                    load_env_cfg(task_id)
                except ValueError:
                    print(f"    Robot: {il_cfg.robot}")
                    print(f"    Scene: {il_cfg.scene}")
                    if il_cfg.task:
                        print(f"    Task:  {il_cfg.task}")

        rl_policies = list_rl_policies(task_id)
        if rl_policies:
            print(f"    RL: {', '.join(rl_policies)}")

        if il_policies:
            print(f"    IL: {', '.join(il_policies)}")

    print()


if __name__ == "__main__":
    main()
