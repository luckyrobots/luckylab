"""Utility for importing all modules in a package recursively."""

from __future__ import annotations

import importlib
import pkgutil
import sys
from collections.abc import Callable, Iterator


def import_packages(package_name: str, blacklist_pkgs: list[str] | None = None) -> None:
    """Import all sub-packages in a package recursively.

    This enables auto-discovery of task packages without manually importing each one.
    Each task package's __init__.py should register itself when imported.

    Args:
        package_name: The package name (e.g., "luckylab.tasks").
        blacklist_pkgs: List of package name patterns to skip.
            Defaults to None (no blacklist).

    Example:
        >>> # In luckylab/tasks/__init__.py
        >>> from luckylab.utils.importer import import_packages
        >>> import_packages(__name__, blacklist_pkgs=["utils", "registry"])
    """
    if blacklist_pkgs is None:
        blacklist_pkgs = []

    # Import the package itself
    package = importlib.import_module(package_name)

    # Import all sub-packages recursively
    for _ in _walk_packages(package.__path__, package.__name__ + ".", blacklist_pkgs=blacklist_pkgs):
        pass


def _walk_packages(
    path: list[str] | None = None,
    prefix: str = "",
    onerror: Callable[[str], None] | None = None,
    blacklist_pkgs: list[str] | None = None,
) -> Iterator:
    """Yield ModuleInfo for all modules recursively on path.

    This is a modified version of pkgutil.walk_packages that adds
    blacklist support for skipping certain packages.

    Args:
        path: List of paths to search.
        prefix: Prefix for module names.
        onerror: Error handler callback.
        blacklist_pkgs: List of package name patterns to skip.

    Yields:
        ModuleInfo for each discovered module.
    """
    if blacklist_pkgs is None:
        blacklist_pkgs = []

    seen: dict[str, bool] = {}

    def mark_seen(p: str) -> bool:
        if p in seen:
            return True
        seen[p] = True
        return False

    for info in pkgutil.iter_modules(path, prefix):
        # Check if blacklisted
        if any(black_pkg in info.name for black_pkg in blacklist_pkgs):
            continue

        yield info

        if info.ispkg:
            try:
                __import__(info.name)
            except Exception:
                if onerror is not None:
                    onerror(info.name)
                else:
                    raise
            else:
                pkg_path = getattr(sys.modules[info.name], "__path__", None) or []
                # Don't traverse paths we've seen before
                pkg_path = [p for p in pkg_path if not mark_seen(p)]
                yield from _walk_packages(pkg_path, info.name + ".", onerror, blacklist_pkgs)
