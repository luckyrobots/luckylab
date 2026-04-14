"""Task contract system for LuckyLab.

Provides TaskContract dataclasses, capability manifest caching,
and contract negotiation with LuckyEngine.
"""

from luckylab.contracts.task_contract import TaskContract
from luckylab.contracts.task_contract import ObservationContract
from luckylab.contracts.task_contract import ActionContract
from luckylab.contracts.task_contract import RewardContract
from luckylab.contracts.task_contract import TerminationContract
from luckylab.contracts.task_contract import RandomizationContract
from luckylab.contracts.task_contract import AuxiliaryDataRequest
from luckylab.contracts.manifest_cache import ManifestCache
from luckylab.contracts.negotiation import negotiate_task, TaskContractError

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
