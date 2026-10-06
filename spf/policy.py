"""Composable direction-only module over an injected baseline and generator."""

from dataclasses import dataclass
from typing import Callable, Sequence

from .waypoint import (
    CONTROLLER_VERSION,
    FORWARD_ACTIONS,
    PROMPT_VERSION,
    CameraGeometry,
    build_waypoint_prompt,
    parse_waypoint,
    waypoint_direction,
)


@dataclass(frozen=True)
class SpfDecision:
    action: int
    baseline: object
    source: str
    waypoint: object = None
    direction: object = None
    raw_output: str | None = None
    fallback_reason: str | None = None

    def to_dict(self):
        return {
            "action": self.action,
            "baseline": self.baseline.to_dict(),
            "source": self.source,
            "waypoint": self.waypoint.to_dict() if self.waypoint else None,
            "direction": self.direction.to_dict() if self.direction else None,
            "raw_output": self.raw_output,
            "fallback_reason": self.fallback_reason,
            "prompt_version": PROMPT_VERSION,
            "controller_version": CONTROLLER_VERSION,
        }


class SpfPolicy:
    def __init__(
        self,
        baseline,
        generate: Callable[[str, Sequence], str],
        action_names: dict[int, str],
        camera: CameraGeometry = CameraGeometry(),
        *,
        enabled: bool = True,
    ):
        self.baseline = baseline
        self.generate = generate
        self.action_names = action_names
        self.camera = camera
        self.enabled = enabled

    def policy(self, instruction, images, actions, *, image_size, pitch_deg=0.0):
        if len(images) == 0:
            raise ValueError("At least one current observation is required")
        base = self.baseline.policy(instruction, images, actions)
        if not self.enabled:
            return SpfDecision(base.action, base, "baseline")
        if base.action == 0:
            return SpfDecision(0, base, "baseline_stop")

        prompt = build_waypoint_prompt(instruction, [self.action_names[a] for a in actions])
        # Model/driver/OOM errors intentionally propagate. Only malformed model
        # output or an explicitly unavailable waypoint falls back to baseline.
        raw = self.generate(prompt, images)
        try:
            point = parse_waypoint(raw)
        except ValueError as exc:
            return SpfDecision(
                base.action, base, "baseline_fallback", raw_output=raw,
                fallback_reason=f"invalid_waypoint: {exc}",
            )
        if not point.visible:
            return SpfDecision(
                base.action, base, "baseline_fallback", waypoint=point,
                raw_output=raw, fallback_reason="waypoint_not_visible",
            )

        # Reuse baseline's relative preference among the existing 3/6/9 m
        # forward primitives. Adaptive distance prediction is a later ablation.
        forward = max(FORWARD_ACTIONS, key=base.probabilities.__getitem__)
        direction = waypoint_direction(
            point, image_size, self.camera, pitch_deg=pitch_deg, forward_action=forward,
        )
        return SpfDecision(direction.action, base, "spf_waypoint", point, direction, raw)
