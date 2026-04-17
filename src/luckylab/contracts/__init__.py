"""Task contract system for LuckyLab.

Provides TaskContract dataclasses, capability manifest caching,
and contract negotiation with LuckyEngine.
"""

from luckylab.contracts.manifest_cache import ManifestCache
from luckylab.contracts.negotiation import TaskContractError, negotiate_task
from luckylab.contracts.task_contract import (
    ActionContract,
    AuxiliaryDataRequest,
    ObservationContract,
    RandomizationContract,
    RewardContract,
    TaskContract,
    TerminationContract,
)

__all__ = [
    "TaskContract",
    "ObservationContract",
    "ActionContract",
    "RewardContract",
    "TerminationContract",
    "RandomizationContract",
    "AuxiliaryDataRequest",
    "ManifestCache",
    "negotiate_task",
    "TaskContractError",
]
