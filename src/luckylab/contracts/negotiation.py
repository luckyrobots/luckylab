"""Contract negotiation with LuckyEngine.

Wraps the luckyrobots client negotiate_task RPC with error handling
and diagnostic formatting.
"""

from __future__ import annotations

import logging
from typing import Any

from luckylab.contracts.task_contract import TaskContract

logger = logging.getLogger(__name__)


class TaskContractError(Exception):
    """Raised when task contract validation fails.

    Attributes:
        errors: List of validation error dicts.
        warnings: List of validation warning dicts.
    """

    def __init__(self, message: str, errors: list[dict] | None = None, warnings: list[dict] | None = None):
        super().__init__(message)
        self.errors = errors or []
        self.warnings = warnings or []

    def format_report(self) -> str:
        """Format a human-readable error report."""
        lines = [str(self)]
        if self.errors:
            lines.append("\nErrors:")
            for err in self.errors:
                lines.append(
                    f"  [{err.get('component', '?')}] {err.get('term_name', '?')}: {err.get('message', '?')}"
                )
                if err.get("suggestion"):
                    lines.append(f"    Suggestion: {err['suggestion']}")
        if self.warnings:
            lines.append("\nWarnings:")
            for warn in self.warnings:
                lines.append(
                    f"  [{warn.get('component', '?')}] {warn.get('term_name', '?')}: "
                    f"{warn.get('message', '?')}"
                )
        return "\n".join(lines)


def negotiate_task(client, contract: TaskContract) -> dict[str, Any]:
    """Negotiate a task contract with the engine.

    Args:
        client: LuckyEngineClient instance (connected).
        contract: TaskContract to negotiate.

    Returns:
        Dict with session_id, reward_terms, termination_terms.

    Raises:
        TaskContractError: If negotiation fails with validation errors.
    """
    contract_dict = contract.to_dict()

    try:
        result = client.negotiate_task(contract_dict)
    except RuntimeError as e:
        raise TaskContractError(str(e)) from e

    if "warnings" in result:
        for w in result["warnings"]:
            logger.warning(
                "[Contract] %s/%s: %s — %s",
                w.get("component", "?"),
                w.get("term_name", "?"),
                w.get("message", "?"),
                w.get("suggestion", ""),
            )

    logger.info(
        "Task contract negotiated: session=%s, rewards=%s, terminations=%s",
        result.get("session_id", ""),
        result.get("reward_terms", []),
        result.get("termination_terms", []),
    )

    return result
