"""CLI: Compare capability manifests between engine versions.

Compares two saved manifests (JSON files) or a saved manifest against
a live engine, showing added, removed, and changed MDP components.

Usage:
    # Compare two saved manifests:
    python -m luckylab.scripts.diff_capabilities --old manifest-v1.json --new manifest-v1.1.json

    # Compare saved manifest against live engine:
    python -m luckylab.scripts.diff_capabilities --old manifest-v1.json --live

    # Save current manifest for later comparison:
    python -m luckylab.scripts.diff_capabilities --save manifest-v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _components_by_name(manifest: dict, category: str) -> dict[str, dict]:
    """Index components by name for a given category."""
    components = manifest.get(category, [])
    return {c["name"]: c for c in components}


def _diff_category(
    category: str,
    old_components: dict[str, dict],
    new_components: dict[str, dict],
) -> list[str]:
    """Diff a single category, returning human-readable change lines."""
    lines = []
    old_names = set(old_components.keys())
    new_names = set(new_components.keys())

    # Added
    for name in sorted(new_names - old_names):
        comp = new_components[name]
        cat = comp.get("category", "")
        desc = comp.get("description", "")
        lines.append(f"  + ADDED:   {name:<35} [{cat}]  {desc}")

    # Removed
    for name in sorted(old_names - new_names):
        comp = old_components[name]
        cat = comp.get("category", "")
        lines.append(f"  - REMOVED: {name:<35} [{cat}]")

    # Changed (present in both — check for differences)
    for name in sorted(old_names & new_names):
        old_comp = old_components[name]
        new_comp = new_components[name]
        changes = _diff_component(old_comp, new_comp)
        if changes:
            lines.append(f"  ~ CHANGED: {name}")
            for change in changes:
                lines.append(f"      {change}")

    return lines


def _diff_component(old: dict, new: dict) -> list[str]:
    """Compare two component dicts, returning list of change descriptions."""
    changes = []

    # Description changed
    if old.get("description", "") != new.get("description", ""):
        changes.append(f"description: {old.get('description', '')!r} -> {new.get('description', '')!r}")

    # Category changed
    if old.get("category", "") != new.get("category", ""):
        changes.append(f"category: {old.get('category', '')} -> {new.get('category', '')}")

    # For randomizations, check range changes
    if "default_range" in old or "default_range" in new:
        old_range = old.get("default_range", (0, 0))
        new_range = new.get("default_range", (0, 0))
        if old_range != new_range:
            changes.append(f"range: {old_range} -> {new_range}")

    if "engine_target" in old or "engine_target" in new:
        old_target = old.get("engine_target", "")
        new_target = new.get("engine_target", "")
        if old_target != new_target:
            changes.append(f"target: {old_target} -> {new_target}")

    return changes


def diff_manifests(old_manifest: dict, new_manifest: dict) -> str:
    """Compare two manifests and return a formatted diff report.

    Args:
        old_manifest: Older manifest dict.
        new_manifest: Newer manifest dict.

    Returns:
        Human-readable diff string.
    """
    lines = []

    old_ver = old_manifest.get("engine_version", "?")
    new_ver = new_manifest.get("engine_version", "?")
    lines.append(f"Comparing: {old_ver} -> {new_ver}")
    lines.append("")

    any_changes = False

    for category in ["observations", "rewards", "terminations"]:
        old_comps = _components_by_name(old_manifest, category)
        new_comps = _components_by_name(new_manifest, category)
        diff_lines = _diff_category(category, old_comps, new_comps)
        if diff_lines:
            any_changes = True
            lines.append(f"{category.upper()}:")
            lines.extend(diff_lines)
            lines.append("")

    # Randomizations have a different structure
    old_rands = {r["name"]: r for r in old_manifest.get("randomizations", [])}
    new_rands = {r["name"]: r for r in new_manifest.get("randomizations", [])}
    rand_lines = _diff_category("randomizations", old_rands, new_rands)
    if rand_lines:
        any_changes = True
        lines.append("RANDOMIZATIONS:")
        lines.extend(rand_lines)
        lines.append("")

    if not any_changes:
        lines.append("No differences found.")

    # Summary
    for category in ["observations", "rewards", "terminations", "randomizations"]:
        if category == "randomizations":
            old_count = len(old_manifest.get(category, []))
            new_count = len(new_manifest.get(category, []))
        else:
            old_count = len(old_manifest.get(category, []))
            new_count = len(new_manifest.get(category, []))
        if old_count != new_count:
            lines.append(f"{category}: {old_count} -> {new_count}")

    return "\n".join(lines)


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="luckylab diff-capabilities",
        description="Compare capability manifests between engine versions.",
    )
    parser.add_argument(
        "--old", help="Path to older manifest JSON file."
    )
    parser.add_argument(
        "--new", help="Path to newer manifest JSON file."
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Use live engine as the 'new' manifest (requires --old)."
    )
    parser.add_argument(
        "--save", help="Save current live manifest to a JSON file and exit."
    )
    parser.add_argument(
        "--robot", default="", help="Robot name filter."
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

    # Mode: save manifest
    if parsed.save:
        manifest = _fetch_live_manifest(parsed)
        with open(parsed.save, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest saved to {parsed.save}")
        return 0

    # Mode: diff two files
    if parsed.old and parsed.new and not parsed.live:
        with open(parsed.old) as f:
            old_manifest = json.load(f)
        with open(parsed.new) as f:
            new_manifest = json.load(f)
        print(diff_manifests(old_manifest, new_manifest))
        return 0

    # Mode: diff old file vs live engine
    if parsed.old and parsed.live:
        with open(parsed.old) as f:
            old_manifest = json.load(f)
        new_manifest = _fetch_live_manifest(parsed)
        print(diff_manifests(old_manifest, new_manifest))
        return 0

    parser.print_help()
    return 1


def _fetch_live_manifest(parsed) -> dict[str, Any]:
    """Connect to engine and fetch capability manifest."""
    from luckyrobots import LuckyEngineClient

    print(f"Connecting to {parsed.host}:{parsed.port}...", file=sys.stderr)
    client = LuckyEngineClient(host=parsed.host, port=parsed.port, timeout=parsed.timeout)
    client.connect()
    if not client.wait_for_server(timeout=parsed.timeout):
        print("ERROR: Could not connect to engine.", file=sys.stderr)
        sys.exit(1)
    try:
        return client.get_capability_manifest(robot_name=parsed.robot)
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
