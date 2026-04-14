"""Task contract dataclasses for declarative MDP specification.

A TaskContract declares what observations, rewards, terminations, and
randomizations a task needs from the engine. The engine validates and
negotiates this contract at session start, providing fail-fast diagnostics
instead of runtime NaN errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ObservationTermRequest:
    """Request for a single observation term."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    group: str = "policy"


@dataclass
class ObservationContract:
    """Declares which observation terms the task requires."""

    required: list[ObservationTermRequest] = field(default_factory=list)
    optional: list[ObservationTermRequest] = field(default_factory=list)


@dataclass
class RewardTermRequest:
    """Request for a single engine-side reward term."""

    name: str
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardContract:
    """Declares reward signals needed from the engine."""

    engine_terms: list[RewardTermRequest] = field(default_factory=list)
    python_terms: list[str] = field(default_factory=list)


@dataclass
class TerminationTermRequest:
    """Request for a single termination condition."""

    name: str
    is_timeout: bool = False
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TerminationContract:
    """Declares termination conditions."""

    terms: list[TerminationTermRequest] = field(default_factory=list)


@dataclass
class ActionTermRequest:
    """Request for a single action term."""

    type: str  # "joint_position", "joint_velocity", "cpg"
    joint_pattern: str = "*"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionContract:
    """Declares the action space the task will use."""

    terms: list[ActionTermRequest] = field(default_factory=list)


@dataclass
class CustomRandomization:
    """A custom domain randomization parameter."""

    name: str
    range_min: float = 0.0
    range_max: float = 1.0
    target: str = ""  # Engine entity or parameter path


@dataclass
class RandomizationContract:
    """Declares domain randomization configuration."""

    custom_randomizations: list[CustomRandomization] = field(default_factory=list)
    # SimulationContract is passed separately via ResetAgentRequest.
    # This contract covers task-specific DR beyond the standard SimulationContract.


@dataclass
class AuxiliaryDataRequest:
    """Request for engine-side data not part of the observation vector."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskContract:
    """Declarative task definition sent from LuckyLab to LuckyEngine at session start.

    This is the canonical way to define a task. Each section maps to one of
    the five existing managers:
    - observations -> ObservationManager
    - rewards -> RewardManager
    - terminations -> TerminationManager
    - randomization -> CurriculumManager + SimulationContract

    Example:
        contract = TaskContract(
            task_id="go2_velocity_flat",
            robot="unitreego2",
            scene="velocity",
            observations=ObservationContract(
                required=[
                    ObservationTermRequest("base_lin_vel"),
                    ObservationTermRequest("base_ang_vel"),
                    ObservationTermRequest("projected_gravity"),
                    ObservationTermRequest("joint_pos"),
                    ObservationTermRequest("joint_vel"),
                    ObservationTermRequest("vel_command"),
                ],
            ),
            rewards=RewardContract(
                engine_terms=[
                    RewardTermRequest("track_linear_velocity", weight=1.0),
                    RewardTermRequest("track_angular_velocity", weight=0.5),
                    RewardTermRequest("feet_air_time", weight=0.2),
                    RewardTermRequest("orientation_error", weight=-0.1),
                ],
            ),
            terminations=TerminationContract(
                terms=[
                    TerminationTermRequest("fell_over"),
                    TerminationTermRequest("time_out", is_timeout=True),
                ],
            ),
        )
    """

    task_id: str = ""
    robot: str = ""
    scene: str = ""
    observations: ObservationContract | None = None
    actions: ActionContract | None = None
    rewards: RewardContract | None = None
    terminations: TerminationContract | None = None
    randomization: RandomizationContract | None = None
    auxiliary_data: list[AuxiliaryDataRequest] = field(default_factory=list)
    max_episode_length_s: float = 20.0

    def to_dict(self) -> dict:
        """Convert to dict for luckyrobots client.negotiate_task()."""
        result: dict = {
            "task_id": self.task_id,
            "robot": self.robot,
            "scene": self.scene,
        }

        if self.observations:
            result["observations"] = {
                "required": [
                    {"name": t.name, "params": t.params, "group": t.group}
                    for t in self.observations.required
                ],
                "optional": [
                    {"name": t.name, "params": t.params, "group": t.group}
                    for t in self.observations.optional
                ],
            }

        if self.rewards:
            result["rewards"] = {
                "engine_terms": [
                    {"name": t.name, "weight": t.weight, "params": t.params}
                    for t in self.rewards.engine_terms
                ],
                "python_terms": self.rewards.python_terms,
            }

        if self.terminations:
            result["terminations"] = {
                "terms": [
                    {
                        "name": t.name,
                        "is_timeout": t.is_timeout,
                        "params": {
                            **t.params,
                            **(
                                {"max_episode_length_s": str(self.max_episode_length_s)}
                                if t.name == "time_out"
                                else {}
                            ),
                        },
                    }
                    for t in self.terminations.terms
                ],
            }

        if self.actions:
            result["actions"] = {
                "terms": [
                    {
                        "type": t.type,
                        "joint_pattern": t.joint_pattern,
                        "params": t.params,
                    }
                    for t in self.actions.terms
                ],
            }

        if self.randomization and self.randomization.custom_randomizations:
            result["randomization"] = {
                "custom_randomizations": [
                    {
                        "name": r.name,
                        "range_min": r.range_min,
                        "range_max": r.range_max,
                        "target": r.target,
                    }
                    for r in self.randomization.custom_randomizations
                ],
            }

        if self.auxiliary_data:
            result["auxiliary_data"] = [
                {"name": a.name, "params": a.params}
                for a in self.auxiliary_data
            ]

        return result
