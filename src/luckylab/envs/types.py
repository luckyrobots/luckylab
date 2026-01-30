"""Type definitions for luckylab environments."""

import numpy as np

# Observation type: dict of observation arrays
VecEnvObs = dict[str, np.ndarray | dict[str, np.ndarray]]

# Step return type: (obs, reward, terminated, truncated, info)
VecEnvStepReturn = tuple[VecEnvObs, float, bool, bool, dict]
