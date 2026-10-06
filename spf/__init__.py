"""SPF-inspired waypoint grounding for the OpenFly Qwen baseline.

The lightweight interfaces do not import torch, transformers, or AirSim.
"""

from .waypoint import CameraGeometry, Waypoint, parse_waypoint, waypoint_direction
from .policy import SpfPolicy

__all__ = ["CameraGeometry", "Waypoint", "parse_waypoint", "waypoint_direction", "SpfPolicy"]
