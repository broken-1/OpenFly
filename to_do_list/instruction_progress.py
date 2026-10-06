"""Independent instruction decomposition and progress tracking prototype.

The module deliberately does not choose navigation actions. A caller supplies a
text-generation function, while this module owns the plan/state JSON contract and
the ordered subgoal state machine.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import re
from typing import Callable, Mapping, Sequence


FINAL_TARGET_CRITERION = (
    "Navigate to within 20 meters of the final target and stop there."
)
INTERMEDIATE_CRITERION_FALLBACK = (
    "Complete this intermediate maneuver or landmark transition, then continue "
    "toward the final target without stopping there."
)


class SubgoalStatus(str, Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"


class TrackerStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class Subgoal:
    id: int
    description: str
    completion_criterion: str
    is_final: bool = False


@dataclass(frozen=True)
class InstructionPlan:
    instruction: str
    subgoals: tuple[Subgoal, ...]
    final_target: str

    @property
    def final_subgoal(self) -> Subgoal:
        return self.subgoals[-1]

    def to_dict(self):
        return {
            "instruction": self.instruction,
            "final_target": self.final_target,
            "subgoals": [asdict(subgoal) for subgoal in self.subgoals],
        }


@dataclass(frozen=True)
class TrackerUpdate:
    subgoal_id: int
    status: TrackerStatus
    progress: float
    confidence: float
    evidence: str


@dataclass(frozen=True)
class ProgressTransition:
    step: int
    subgoal_id: int
    advanced: bool
    completed: bool
    plan_finished: bool


def _load_payload(payload: str | Mapping) -> Mapping:
    if isinstance(payload, str):
        payload = json.loads(payload)
    return payload


def _normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _remove_final_target_from_description(
    description: str,
    final_target: str,
) -> str:
    """Keep a useful maneuver prefix but remove a duplicated final arrival."""
    description_words = [
        word
        for word in _normalized_words(description)
        if word not in {"a", "an", "the"}
    ]
    target_words = [
        word
        for word in _normalized_words(final_target)
        if word not in {"a", "an", "the"}
    ]
    if not target_words:
        return description.strip()
    for index in range(len(description_words) - len(target_words) + 1):
        if description_words[index : index + len(target_words)] != target_words:
            continue
        prefix = description_words[:index]
        while prefix and prefix[-1] in {"to", "toward", "the", "and"}:
            prefix.pop()
        if len(prefix) < 2:
            return ""
        return " ".join(prefix)
    return description.strip()


def build_planner_prompt(instruction: str) -> str:
    """Build the one-shot prompt used to decompose the original instruction."""
    return f"""
You are the instruction planner for a visual-language navigation task.

Original instruction:
{instruction}

Split the instruction into a short ordered list of observable navigation subgoals.
Preserve the order and do not invent landmarks or actions that are not supported by
the instruction. Each subgoal must describe one meaningful stage, not one low-level
drone action. Reserve the final arrival at the destination for the last subgoal only:
earlier subgoals must not repeat the final target or claim that the final destination
has been reached. The final subgoal must describe the final target, and its completion
criterion must require reaching within 20 meters of that target and stopping there.

Return only valid JSON with this schema:
{{
  "final_target": "short description of the final target",
  "subgoals": [
    {{
      "id": 1,
      "description": "first ordered subgoal",
      "completion_criterion": "observable evidence that this subgoal is complete"
    }}
  ]
}}
""".strip()


def parse_instruction_plan(payload: str | Mapping, instruction: str) -> InstructionPlan:
    """Parse a planner response and ensure the final target is an ordered subgoal."""
    data = _load_payload(payload)
    raw_subgoals = data["subgoals"]
    final_target_data = data["final_target"]
    final_target = (
        final_target_data["description"]
        if isinstance(final_target_data, Mapping)
        else str(final_target_data)
    ).strip()

    subgoals = []
    for raw_subgoal in raw_subgoals:
        description = str(raw_subgoal["description"]).strip()
        criterion = str(raw_subgoal["completion_criterion"]).strip()
        cleaned_description = _remove_final_target_from_description(
            description,
            final_target,
        )
        if not cleaned_description:
            continue
        if any(
            marker in criterion.lower()
            for marker in ("20 meter", "20-meter", "stop at", "stop there")
        ):
            criterion = INTERMEDIATE_CRITERION_FALLBACK
        subgoals.append(
            Subgoal(
                id=len(subgoals) + 1,
                description=cleaned_description,
                completion_criterion=criterion,
            )
        )

    subgoals.append(
        Subgoal(
            id=len(subgoals) + 1,
            description=final_target,
            completion_criterion=FINAL_TARGET_CRITERION,
        )
    )

    subgoals[-1] = Subgoal(
        id=subgoals[-1].id,
        description=subgoals[-1].description,
        completion_criterion=subgoals[-1].completion_criterion,
        is_final=True,
    )
    return InstructionPlan(
        instruction=instruction,
        subgoals=tuple(subgoals),
        final_target=final_target,
    )


def build_tracker_prompt(
    state: "ProgressState",
    step: int,
    recent_actions: Sequence[str],
) -> str:
    """Build the per-observation prompt for the currently active subgoal."""
    active = state.active_subgoal
    completed = [
        f"{subgoal.id}. {subgoal.description}"
        for subgoal, status in zip(state.plan.subgoals, state.statuses)
        if status == SubgoalStatus.COMPLETED
    ]
    completed_text = "none" if not completed else "; ".join(completed)
    return f"""
You are the progress tracker for a visual-language navigation task.

Original instruction:
{state.plan.instruction}

Final target:
{state.plan.final_target}

Current navigation step: {step}
Currently executing subgoal {active.id} of {len(state.plan.subgoals)}:
{active.description}

Completion criterion for the current subgoal:
{active.completion_criterion}

Subgoals already completed:
{completed_text}

Recent executed actions:
{json.dumps(list(recent_actions))}

Inspect the current visual observation and decide only the status of the current
subgoal. Do not skip to a later subgoal. Use IN_PROGRESS when evidence is insufficient,
COMPLETED only when the criterion is visually satisfied, and BLOCKED when progress is
impossible or the instruction is inconsistent with the observation. The progress score
is an estimate from 0.0 to 1.0 for this subgoal, not the probability of final task
success. A landmark being absent from the current view means IN_PROGRESS, not BLOCKED.
For the final subgoal, COMPLETED means the drone appears to be within roughly 20 meters
of the described final target and should stop now; the controller will issue STOP.

Return only valid JSON:
{{
  "subgoal_id": {active.id},
  "status": "IN_PROGRESS|COMPLETED|BLOCKED",
  "progress": 0.0,
  "confidence": 0.0,
  "evidence": "brief visual evidence"
}}
""".strip()


def build_action_prompt(
    state: "ProgressState",
    recent_actions: Sequence[str],
    action_space: Mapping[int, str],
) -> str:
    """Build the navigation prompt conditioned on the active ordered subgoal."""
    active = state.active_subgoal
    completed = [
        subgoal.description
        for subgoal, status in zip(state.plan.subgoals, state.statuses)
        if status == SubgoalStatus.COMPLETED
    ]
    action_text = ", ".join(
        f"{action}:{name}" for action, name in action_space.items()
    )
    return f"""
You control a drone in a city visual-language navigation task.

Original instruction:
{state.plan.instruction}

Final target:
{state.plan.final_target}

Completed subgoals:
{json.dumps(completed)}

Currently executing subgoal {active.id} of {len(state.plan.subgoals)}:
{active.description}

Current subgoal completion criterion:
{active.completion_criterion}

Recent executed actions:
{json.dumps(list(recent_actions))}

Action space:
{action_text}

Choose the single best next movement action for the current subgoal. Preserve the order
of the original instruction and do not navigate directly to a later subgoal. Prefer
long forward motion only when the route is visually clear and aligned; use short motion
near landmarks, turns, obstacles, and the final target. STOP is controlled separately
by the progress tracker and is not available here. Respond with exactly one action ID.
""".strip()


def parse_tracker_update(payload: str | Mapping) -> TrackerUpdate:
    data = _load_payload(payload)
    status = TrackerStatus(str(data["status"]).upper())
    progress = float(data["progress"])
    confidence = float(data["confidence"])
    if not 0.0 <= progress <= 1.0:
        raise ValueError("tracker progress must be between 0 and 1")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("tracker confidence must be between 0 and 1")
    return TrackerUpdate(
        subgoal_id=int(data["subgoal_id"]),
        status=status,
        progress=progress,
        confidence=confidence,
        evidence=str(data.get("evidence", "")).strip(),
    )


@dataclass
class ProgressState:
    plan: InstructionPlan
    active_index: int = 0
    statuses: list[SubgoalStatus] = field(default_factory=list)
    progress: list[float] = field(default_factory=list)
    confidence: float = 0.0
    last_evidence: str = ""
    last_step: int | None = None

    def __post_init__(self):
        count = len(self.plan.subgoals)
        if not self.statuses:
            self.statuses = [SubgoalStatus.PENDING] * count
            self.statuses[0] = SubgoalStatus.ACTIVE
        if not self.progress:
            self.progress = [0.0] * count

    @property
    def active_subgoal(self) -> Subgoal:
        return self.plan.subgoals[self.active_index]

    @property
    def finished(self) -> bool:
        return self.active_index >= len(self.plan.subgoals)

    def apply(
        self,
        update: TrackerUpdate,
        step: int,
        completion_confidence_threshold: float = 0.75,
    ) -> ProgressTransition:
        if self.finished:
            raise RuntimeError("instruction plan is already finished")
        active = self.active_subgoal
        if update.subgoal_id != active.id:
            raise ValueError(
                f"tracker updated subgoal {update.subgoal_id}, "
                f"but subgoal {active.id} is active"
            )

        self.progress[self.active_index] = update.progress
        self.confidence = update.confidence
        self.last_evidence = update.evidence
        self.last_step = step
        completed = (
            update.status == TrackerStatus.COMPLETED
            and update.confidence >= completion_confidence_threshold
        )
        if completed:
            self.statuses[self.active_index] = SubgoalStatus.COMPLETED
            self.active_index += 1
            if not self.finished:
                self.statuses[self.active_index] = SubgoalStatus.ACTIVE
            return ProgressTransition(
                step=step,
                subgoal_id=active.id,
                advanced=True,
                completed=True,
                plan_finished=self.finished,
            )

        self.statuses[self.active_index] = (
            SubgoalStatus.BLOCKED
            if update.status == TrackerStatus.BLOCKED
            else SubgoalStatus.ACTIVE
        )
        return ProgressTransition(
            step=step,
            subgoal_id=active.id,
            advanced=False,
            completed=False,
            plan_finished=False,
        )

    def to_dict(self):
        return {
            "plan": self.plan.to_dict(),
            "active_index": self.active_index,
            "statuses": [status.value for status in self.statuses],
            "progress": self.progress,
            "confidence": self.confidence,
            "last_evidence": self.last_evidence,
            "last_step": self.last_step,
        }


class InstructionProgressController:
    """Glue the Qwen text generator to the deterministic progress state machine."""

    def __init__(
        self,
        generate: Callable[[str, Sequence], str],
        completion_confidence_threshold: float = 0.75,
    ):
        self.generate = generate
        self.completion_confidence_threshold = completion_confidence_threshold

    def create_plan(self, instruction: str) -> ProgressState:
        payload = self.generate(build_planner_prompt(instruction), ())
        plan = parse_instruction_plan(payload, instruction)
        return ProgressState(plan)

    def update(
        self,
        state: ProgressState,
        step: int,
        images: Sequence,
        recent_actions: Sequence[str],
    ) -> ProgressTransition:
        prompt = build_tracker_prompt(state, step, recent_actions)
        payload = self.generate(prompt, images)
        update = parse_tracker_update(payload)
        return state.apply(
            update,
            step=step,
            completion_confidence_threshold=self.completion_confidence_threshold,
        )
