"""CLI: List available MDP capabilities from the engine.

Connects to a running LuckyEngine instance and displays all available
observations, rewards, terminations, and randomizations that can be
used in a TaskContract.

Usage:
    python -m luckylab.scripts.list_capabilities
    python -m luckylab.scripts.list_capabilities --robot unitreego2
    python -m luckylab.scripts.list_capabilities --host 192.168.1.10 --port 50051
"""

from __future__ import annotations

import argparse
import sys

from luckyrobots import LuckyEngineClient

from luckylab.contracts.manifest_cache import ManifestCache


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="luckylab list-capabilities",
        description="List available MDP capabilities from a running LuckyEngine instance.",
    )
    parser.add_argument(
        "--robot", default="", help="Filter by robot name (e.g., unitreego2). Empty = show all."
    )
    parser.add_argument(
        "--scene", default="", help="Filter by scene name. Empty = show all."
    )
    parser.add_argument(
        "--host", default="localhost", help="Engine gRPC host (default: localhost)."
    )
    parser.add_argument(
        "--port", type=int, default=50051, help="Engine gRPC port (default: 50051)."
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="Connection timeout in seconds."
    )
    parsed = parser.parse_args(args)

    print(f"Connecting to LuckyEngine at {parsed.host}:{parsed.port}...")

    try:
        client = LuckyEngineClient(host=parsed.host, port=parsed.port, timeout=parsed.timeout)
        client.connect()
        if not client.wait_for_server(timeout=parsed.timeout):
            print("ERROR: Could not connect to engine. Is it running?", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"ERROR: Connection failed: {e}", file=sys.stderr)
        return 1

    try:
        cache = ManifestCache(ttl_seconds=60)
        manifest = cache.get(client, robot=parsed.robot, scene=parsed.scene)
        output = cache.print_manifest(manifest)
        print(output)

        # Print robot info if available
        robot_info = manifest.get("robot_info")
        if robot_info:
            print(f"{'=' * 60}")
            print(f"  ROBOT INFO")
            print(f"{'=' * 60}")
            print(f"  Name: {robot_info.get('name', '?')}")
            joints = robot_info.get("joint_names", [])
            if joints:
                print(f"  Joints ({len(joints)}): {', '.join(joints)}")

        # Print summary
        obs_count = len(manifest.get("observations", []))
        rew_count = len(manifest.get("rewards", []))
        term_count = len(manifest.get("terminations", []))
        rand_count = len(manifest.get("randomizations", []))
        print(f"\nTotal: {obs_count} observations, {rew_count} rewards, "
              f"{term_count} terminations, {rand_count} randomizations")

    except Exception as e:
        print(f"ERROR: Failed to fetch manifest: {e}", file=sys.stderr)
        return 1
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
