"""CLI: Validate a task contract against engine capabilities.

Connects to a running LuckyEngine instance, loads a registered task's
contract, validates it, and reports errors/warnings without starting training.

Usage:
    python -m luckylab.scripts.validate_task go2_velocity_flat
    python -m luckylab.scripts.validate_task go2_velocity_flat --host localhost --port 50051
"""

from __future__ import annotations

import argparse
import sys

from luckyrobots import LuckyEngineClient


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="luckylab validate-task",
        description="Validate a task contract against engine capabilities.",
    )
    parser.add_argument(
        "task_id",
        help="Task ID to validate (must be registered in luckylab task registry).",
    )
    parser.add_argument(
        "--host", default="localhost", help="Engine gRPC host."
    )
    parser.add_argument(
        "--port", type=int, default=50051, help="Engine gRPC port."
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="Connection timeout."
    )
    parsed = parser.parse_args(args)

    # Import tasks to populate registry
    try:
        import luckylab.tasks  # noqa: F401
    except ImportError:
        pass

    from luckylab.tasks.registry import load_env_cfg

    # Load task env config
    env_cfg = load_env_cfg(parsed.task_id)
    if env_cfg is None:
        print(f"ERROR: Task '{parsed.task_id}' not found in registry.", file=sys.stderr)
        print("Available tasks:", file=sys.stderr)
        from luckylab.tasks.registry import _REGISTRY
        for name in _REGISTRY:
            print(f"  - {name}", file=sys.stderr)
        return 1

    # Check for task contract
    task_contract = getattr(env_cfg, "task_contract", None)
    if task_contract is None:
        print(f"Task '{parsed.task_id}' does not have a TaskContract defined.")
        print("The task uses the existing manager-based pipeline (no engine-side MDP).")
        print("To add a contract, set task_contract in the env config.")
        return 0

    # Connect and validate
    print(f"Connecting to LuckyEngine at {parsed.host}:{parsed.port}...")
    try:
        client = LuckyEngineClient(host=parsed.host, port=parsed.port, timeout=parsed.timeout)
        client.connect()
        if not client.wait_for_server(timeout=parsed.timeout):
            print("ERROR: Could not connect to engine.", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"ERROR: Connection failed: {e}", file=sys.stderr)
        return 1

    try:
        contract_dict = task_contract.to_dict()
        result = client.negotiate_task(contract_dict)

        print(f"\nValidation PASSED for '{parsed.task_id}'")
        print(f"  Session ID: {result.get('session_id', '?')}")
        print(f"  Engine reward terms: {result.get('reward_terms', [])}")
        print(f"  Engine termination terms: {result.get('termination_terms', [])}")

        warnings = result.get("warnings", [])
        if warnings:
            print(f"\n  Warnings ({len(warnings)}):")
            for w in warnings:
                print(f"    [{w.get('component', '?')}] {w.get('term_name', '?')}: {w.get('message', '?')}")
                if w.get("suggestion"):
                    print(f"      Suggestion: {w['suggestion']}")

        return 0

    except RuntimeError as e:
        print(f"\nValidation FAILED for '{parsed.task_id}':")
        print(f"  {e}")
        return 1
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
