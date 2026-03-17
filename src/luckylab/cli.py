"""LuckyLab CLI — unified entry point for training and inference."""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: lucky <command> [args...]")
        print()
        print("Commands:")
        print("  train      Train a policy (RL or IL)")
        print("  play       Run inference with a trained policy")
        print()
        print("Examples:")
        print("  lucky play piper_blockstacking --policy act --checkpoint runs/.../final")
        print("  lucky train go2_velocity_flat --algorithm ppo")
        sys.exit(1)

    command = sys.argv[1]

    # Remove the command name so the subcommand sees clean argv
    sys.argv = [f"lucky {command}"] + sys.argv[2:]

    if command == "play":
        from luckylab.scripts.play import main as play_main
        sys.exit(play_main())
    elif command == "train":
        from luckylab.scripts.train import main as train_main
        sys.exit(train_main())
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, play")
        sys.exit(1)


if __name__ == "__main__":
    main()
