"""Type definitions for luckylab environments."""

from typing import Dict

import numpy as np

# Observation type: dict of observation arrays
VecEnvObs = Dict[str, np.ndarray | Dict[str, np.ndarray]]

# Step return type: (obs, reward, terminated, truncated, info)
VecEnvStepReturn = tuple[VecEnvObs, float, bool, bool, dict]
