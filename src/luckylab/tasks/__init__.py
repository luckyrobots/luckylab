"""LuckyLab task modules.

Tasks are auto-discovered and registered when this module is imported.
Each task package registers itself in its __init__.py.
"""

from ..utils.importer import import_packages

_BLACKLIST_PKGS = ["registry", ".mdp"]
import_packages(__name__, _BLACKLIST_PKGS)
