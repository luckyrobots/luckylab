"""Capability manifest cache with TTL-based refresh.

Caches the engine's capability manifest (available observations, rewards,
terminations, randomizations) to avoid repeated gRPC calls during task
configuration and validation.

Usage:
    cache = ManifestCache(ttl_seconds=300)
    manifest = cache.get(client, robot="unitreego2")
    # Subsequent calls within TTL return cached result.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class ManifestCache:
    """TTL-based cache for engine capability manifests.

    The manifest describes what MDP components the engine supports.
    Since components don't change during a training session, caching
    avoids redundant GetCapabilityManifest RPCs.

    Attributes:
        ttl_seconds: Cache lifetime in seconds. Default 300 (5 minutes).
    """

    def __init__(self, ttl_seconds: float = 300.0):
        self._ttl_seconds = ttl_seconds
        self._cache: dict[str, _CacheEntry] = {}

    def get(
        self,
        client,
        robot: str = "",
        scene: str = "",
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Get the capability manifest, using cache if available.

        Args:
            client: LuckyEngineClient instance (connected).
            robot: Robot name filter.
            scene: Scene name filter.
            force_refresh: Bypass cache and fetch fresh manifest.

        Returns:
            Dict with observations, rewards, terminations, randomizations lists.
        """
        cache_key = f"{robot}:{scene}"

        if not force_refresh:
            entry = self._cache.get(cache_key)
            if entry is not None and not entry.is_expired(self._ttl_seconds):
                return entry.manifest

        manifest = client.get_capability_manifest(robot_name=robot, scene=scene)
        self._cache[cache_key] = _CacheEntry(manifest=manifest, timestamp=time.monotonic())

        logger.debug(
            "Manifest cached for %s: %d obs, %d rewards, %d terms, %d rand",
            cache_key,
            len(manifest.get("observations", [])),
            len(manifest.get("rewards", [])),
            len(manifest.get("terminations", [])),
            len(manifest.get("randomizations", [])),
        )

        return manifest

    def invalidate(self, robot: str = "", scene: str = ""):
        """Invalidate a specific cache entry or all entries.

        Args:
            robot: Robot name. If empty with scene empty, clears all.
            scene: Scene name.
        """
        if not robot and not scene:
            self._cache.clear()
        else:
            cache_key = f"{robot}:{scene}"
            self._cache.pop(cache_key, None)

    def refresh(self, client, robot: str = "", scene: str = "") -> dict[str, Any]:
        """Force-refresh the manifest for a robot/scene combination.

        Shorthand for get(..., force_refresh=True).
        """
        return self.get(client, robot=robot, scene=scene, force_refresh=True)

    def print_manifest(self, manifest: dict[str, Any]) -> str:
        """Format a manifest as a human-readable table.

        Returns:
            Formatted string with sections for each component category.
        """
        lines = []

        lines.append(
            f"Engine: {manifest.get('engine_version', '?')} "
            f"(manifest v{manifest.get('manifest_version', '?')})"
        )
        lines.append("")

        for category in ["observations", "rewards", "terminations"]:
            components = manifest.get(category, [])
            if not components:
                continue
            lines.append(f"{'=' * 60}")
            lines.append(f"  {category.upper()} ({len(components)} available)")
            lines.append(f"{'=' * 60}")
            for comp in components:
                name = comp.get("name", "?")
                desc = comp.get("description", "")
                cat = comp.get("category", "")
                lines.append(f"  {name:<35} [{cat}]")
                if desc:
                    lines.append(f"    {desc}")
            lines.append("")

        randomizations = manifest.get("randomizations", [])
        if randomizations:
            lines.append(f"{'=' * 60}")
            lines.append(f"  RANDOMIZATIONS ({len(randomizations)} available)")
            lines.append(f"{'=' * 60}")
            for r in randomizations:
                name = r.get("name", "?")
                desc = r.get("description", "")
                rng = r.get("default_range", (0, 0))
                target = r.get("engine_target", "")
                lines.append(f"  {name:<35} [{rng[0]:.2f}, {rng[1]:.2f}] → {target}")
                if desc:
                    lines.append(f"    {desc}")
            lines.append("")

        return "\n".join(lines)


class _CacheEntry:
    """Internal cache entry with timestamp."""

    __slots__ = ("manifest", "timestamp")

    def __init__(self, manifest: dict[str, Any], timestamp: float):
        self.manifest = manifest
        self.timestamp = timestamp

    def is_expired(self, ttl_seconds: float) -> bool:
        return (time.monotonic() - self.timestamp) > ttl_seconds
