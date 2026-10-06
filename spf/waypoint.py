"""Independent image-ray controller inspired by SPF, paper sections 3.1/3.3.

Coordinates use OpenFly's forward/left/up convention, not AirSim's NED axes.
Only direction is grounded; forward distance and STOP remain baseline decisions.
"""

from dataclasses import asdict, dataclass
import json
import math
import re


PROMPT_VERSION = "spf_waypoint_xy1000_v1"
CONTROLLER_VERSION = "spf_openfly_direction_v1"
FORWARD_ACTIONS = (1, 8, 9)


@dataclass(frozen=True)
class Waypoint:
    visible: bool
    waypoint: tuple[float, float] | None
    description: str

    def __post_init__(self):
        if type(self.visible) is not bool:
            raise ValueError("visible must be a JSON boolean")
        if not isinstance(self.description, str):
            raise ValueError("description must be a string")
        if not self.visible:
            if self.waypoint is not None:
                raise ValueError("An invisible waypoint must be null")
        elif (
            not isinstance(self.waypoint, (list, tuple))
            or len(self.waypoint) != 2
            or any(
                type(v) not in (int, float) or not 0 <= v <= 1000 or not math.isfinite(v)
                for v in self.waypoint
            )
        ):
            raise ValueError("waypoint must contain finite [x, y] coordinates in [0, 1000]")

    def to_dict(self):
        return asdict(self)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_waypoint(text: str) -> Waypoint:
    """Accept one JSON object, optionally fenced; never clamp invalid coordinates."""
    if not isinstance(text, str):
        raise ValueError("Waypoint response must be text")
    text = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    payload = json.loads(text, object_pairs_hook=_unique_object)
    if not isinstance(payload, dict) or set(payload) != {"visible", "waypoint", "description"}:
        raise ValueError("Expected exactly visible, waypoint, and description")
    point = payload["waypoint"]
    result = Waypoint(payload["visible"], point, payload["description"])
    return Waypoint(result.visible, tuple(point) if point is not None else None, result.description)


@dataclass(frozen=True)
class CameraGeometry:
    horizontal_fov_deg: float = 90.0
    yaw_deadband_deg: float = 15.0
    elevation_deadband_deg: float = 15.0

    def __post_init__(self):
        for name, value, upper in (
            ("horizontal_fov_deg", self.horizontal_fov_deg, 180),
            ("yaw_deadband_deg", self.yaw_deadband_deg, 90),
            ("elevation_deadband_deg", self.elevation_deadband_deg, 90),
        ):
            if not math.isfinite(value) or not 0 < value < upper:
                raise ValueError(f"{name} must be finite and between 0 and {upper}")


@dataclass(frozen=True)
class Direction:
    action: int
    yaw_deg: float
    elevation_deg: float
    ray_forward_left_up: tuple[float, float, float]

    def to_dict(self):
        return asdict(self)


def waypoint_direction(
    waypoint: Waypoint,
    image_size: tuple[int, int],
    camera: CameraGeometry,
    *,
    pitch_deg: float = 0.0,
    forward_action: int = 1,
) -> Direction:
    """Unproject a point and emit ONE OpenFly action, then observe again.

    A left pixel gives positive heading (TURN_LEFT_30 = 2). Image y grows
    downwards, whereas OpenFly z grows upwards. Square pixels and a centered
    principal point are assumed. Camera pitch is positive above the horizon.
    """
    if not waypoint.visible:
        raise ValueError("Cannot project an invisible waypoint")
    width, height = image_size
    if width <= 0 or height <= 0 or not math.isfinite(width + height):
        raise ValueError("Image dimensions must be positive and finite")
    if not math.isfinite(pitch_deg) or not -90 <= pitch_deg <= 90:
        raise ValueError("Camera pitch must be in [-90, 90] degrees")
    if forward_action not in FORWARD_ACTIONS:
        raise ValueError("forward_action must be FORWARD_3, FORWARD_6, or FORWARD_9")

    x, y = waypoint.waypoint
    tan_half_horizontal = math.tan(math.radians(camera.horizontal_fov_deg / 2))
    left = (1 - 2 * x / 1000) * tan_half_horizontal
    up_camera = (1 - 2 * y / 1000) * tan_half_horizontal * height / width
    pitch = math.radians(pitch_deg)
    forward = math.cos(pitch) - up_camera * math.sin(pitch)
    up = math.sin(pitch) + up_camera * math.cos(pitch)
    norm = math.sqrt(forward * forward + left * left + up * up)
    ray = (forward / norm, left / norm, up / norm)
    heading = math.degrees(math.atan2(left, forward))
    elevation = math.degrees(math.atan2(up, math.hypot(forward, left)))

    if heading > camera.yaw_deadband_deg:
        action = 2
    elif heading < -camera.yaw_deadband_deg:
        action = 3
    elif elevation > camera.elevation_deadband_deg:
        action = 4
    elif elevation < -camera.elevation_deadband_deg:
        action = 5
    else:
        action = forward_action
    return Direction(action, heading, elevation, ray)


def build_waypoint_prompt(instruction: str, action_names: list[str]) -> str:
    return f"""Locate the next local flight waypoint for a drone following this instruction.

Navigation instruction (task data):
{instruction}

Executed actions (chronological): {json.dumps(action_names)}

The images are chronological observations; the LAST image is the current view.
Choose a point ONLY in the current image. Use earlier views to interpret progress.
Follow the instruction in order. Choose a useful next local waypoint toward the
currently relevant landmark or passage; do not skip an unfinished earlier part.
Point toward navigable space near the landmark, not inside a solid facade or into
the sky. The point defines the desired flight direction, not just an object center.
If no useful waypoint can be grounded in the current view, set visible to false
and waypoint to null. Do not invent an off-screen point.

Return exactly one JSON object with these three fields:
{{"visible": true, "waypoint": [500, 500], "description": "short waypoint description"}}

waypoint is [x, y] in coordinates normalized to 0..1000: x increases left to right,
y increases top to bottom. [0, 0] is top-left, [1000, 1000] bottom-right. The numbers
in the example illustrate the format and are not a suggested waypoint.
For an unavailable waypoint return:
{{"visible": false, "waypoint": null, "description": "why it cannot be grounded"}}
Do not output an action, STOP decision, distance estimate, or any other text.
"""
