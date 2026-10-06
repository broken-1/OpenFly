"""CPU-only protocol, camera geometry, ablation, and evaluation regression tests."""

import ast
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from spf.evaluate import parse_args, prepare_run, run_episode
from spf.policy import SpfPolicy
from spf.waypoint import CameraGeometry, Waypoint, parse_waypoint, waypoint_direction


ROOT = Path(__file__).resolve().parents[1]


def load_project_functions():
    """Use actual baseline kinematics/scoring without importing GPU/sim packages."""
    tree = ast.parse((ROOT / "train/eval.py").read_text(encoding="utf-8"))
    names = {"calculate_distance", "getPoseAfterMakeAction"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(functions) != len(names):
        raise AssertionError("Baseline movement/metric function interface changed")
    namespace = {"math": math}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "train/eval.py", "exec"), namespace)
    return namespace["getPoseAfterMakeAction"], namespace["calculate_distance"]


class BaseOutput:
    def __init__(self, action, forward=9):
        self.action = action
        scores = [0.01] * 10
        scores[forward] = 0.2
        scores[action] = 0.6
        self.probabilities = tuple(v / sum(scores) for v in scores)

    def to_dict(self):
        return {"action": self.action, "probabilities": self.probabilities}


ACTION_NAMES = {i: str(i) for i in range(10)}
VISIBLE = json.dumps({"visible": True, "waypoint": [500, 500], "description": "passage"})


class WaypointTests(unittest.TestCase):
    def test_json_and_fenced_json(self):
        expected = Waypoint(True, (500, 500), "passage")
        self.assertEqual(parse_waypoint(VISIBLE), expected)
        self.assertEqual(parse_waypoint(f"```json\n{VISIBLE}\n```"), expected)

    def test_invalid_grounding_is_rejected_without_clamping(self):
        for point in ([-1, 500], [1001, 500], [True, 500], [float("nan"), 500],
                      [float("inf"), 500], [10**400, 500], [500], "500,500", None):
            with self.subTest(point=point), self.assertRaises(ValueError):
                parse_waypoint(json.dumps({"visible": True, "waypoint": point, "description": "x"}))

    def test_ambiguous_schema_is_rejected(self):
        for text in (
            '{"visible":true,"visible":false,"waypoint":null,"description":"x"}',
            '{"visible":"false","waypoint":null,"description":"x"}',
            '{"visible":false,"waypoint":[500,500],"description":"x"}',
            '{"visible":true,"waypoint":[500,500],"description":"x","STOP":true}',
            VISIBLE + VISIBLE, "[]", "truncated {",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_waypoint(text)


class GeometryTests(unittest.TestCase):
    def direction(self, xy, **kwargs):
        return waypoint_direction(Waypoint(True, xy, "test"), (1920, 1080), CameraGeometry(), **kwargs)

    def test_center_uses_baseline_forward_length(self):
        for action in (1, 8, 9):
            self.assertEqual(self.direction((500, 500), forward_action=action).action, action)

    def test_image_directions_match_actual_openfly_coordinates(self):
        move, _ = load_project_functions()
        for point, expected in (((0, 500), 2), ((1000, 500), 3), ((500, 0), 4), ((500, 1000), 5)):
            with self.subTest(point=point):
                decision = self.direction(point)
                self.assertEqual(decision.action, expected)
                pose = move([0, 0, 0, 0], decision.action)
                if expected in (2, 3):
                    self.assertGreater(pose[3] * decision.yaw_deg, 0)
                else:
                    self.assertGreater(pose[2] * decision.elevation_deg, 0)

    def test_pitch_compensation(self):
        self.assertEqual(self.direction((500, 500), pitch_deg=-45).action, 5)
        self.assertEqual(self.direction((500, 500), pitch_deg=45).action, 4)
        self.assertAlmostEqual(self.direction((500, 500), pitch_deg=-45).elevation_deg, -45)

    def test_aspect_ratio_and_resolution(self):
        point = Waypoint(True, (500, 250), "test")
        wide = waypoint_direction(point, (1920, 1080), CameraGeometry())
        scaled = waypoint_direction(point, (960, 540), CameraGeometry())
        square = waypoint_direction(point, (1080, 1080), CameraGeometry())
        self.assertEqual(wide, scaled)
        self.assertGreater(square.elevation_deg, wide.elevation_deg)
        self.assertAlmostEqual(sum(v * v for v in wide.ray_forward_left_up), 1)

    def test_no_camera_calibration_or_invalid_pitch_does_not_silently_fallback(self):
        for fov in (0, 180, float("nan")):
            with self.subTest(fov=fov), self.assertRaises(ValueError):
                CameraGeometry(horizontal_fov_deg=fov)
        with self.assertRaises(ValueError):
            self.direction((500, 500), pitch_deg=float("nan"))


class PolicyTests(unittest.TestCase):
    def make_policy(self, action=1, text=VISIBLE, enabled=True):
        base = Mock()
        base.policy.return_value = BaseOutput(action)
        generate = Mock(return_value=text)
        return SpfPolicy(base, generate, ACTION_NAMES, enabled=enabled), base, generate

    def call_policy(self, policy):
        return policy.policy("go to the passage", [object()], [1], image_size=(1920, 1080))

    def test_disabled_module_preserves_all_baseline_actions(self):
        for action in range(10):
            policy, base, generate = self.make_policy(action, enabled=False)
            decision = self.call_policy(policy)
            self.assertIs(decision.baseline, base.policy.return_value)
            self.assertEqual(decision.action, action)
            generate.assert_not_called()

    def test_baseline_stop_is_preserved_without_grounding_call(self):
        policy, _, generate = self.make_policy(action=0)
        decision = self.call_policy(policy)
        self.assertEqual((decision.action, decision.source), (0, "baseline_stop"))
        generate.assert_not_called()

    def test_grounded_direction_uses_same_observations_and_logs_original_probabilities(self):
        policy, base, generate = self.make_policy(
            action=9, text='{"visible":true,"waypoint":[0,500],"description":"left passage"}',
        )
        images, actions = [object(), object()], [1, 2]
        decision = policy.policy("instruction", images, actions, image_size=(1920, 1080))
        base.policy.assert_called_once_with("instruction", images, actions)
        self.assertIs(generate.call_args.args[1], images)
        self.assertEqual(decision.action, 2)
        self.assertEqual(decision.to_dict()["baseline"]["action"], 9)
        self.assertNotIn("probabilities", decision.to_dict())

    def test_invisible_or_invalid_waypoint_falls_back(self):
        for text in ("invalid", '{"visible":false,"waypoint":null,"description":"outside view"}'):
            policy, _, _ = self.make_policy(action=7, text=text)
            decision = self.call_policy(policy)
            self.assertEqual((decision.action, decision.source), (7, "baseline_fallback"))
            self.assertTrue(decision.fallback_reason)

    def test_model_failures_propagate(self):
        policy, _, generate = self.make_policy()
        generate.side_effect = RuntimeError("CUDA out of memory")
        with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
            self.call_policy(policy)


class EvaluationTests(unittest.TestCase):
    def episode(self, actions, goal, history_images=3):
        move, distance = load_project_functions()
        base = Mock()
        base.policy.side_effect = [BaseOutput(a) for a in actions]
        policy = SpfPolicy(base, Mock(), ACTION_NAMES, enabled=False)
        image = SimpleNamespace(shape=(1080, 1920, 3))
        bridge = Mock()
        bridge.get_camera_data.return_value = image
        item = {"image_path": "env_airsim_23/low", "gpt_instruction": "navigate",
                "pos": [[0, 0, 0], goal], "yaw": [0]}
        records = []
        with redirect_stdout(io.StringIO()):
            result = run_episode(bridge, policy, item, 0, len(actions), history_images,
                                 ACTION_NAMES, move, distance, records.append)
        return result, records, base

    def test_last_movement_counts_even_without_stop(self):
        summary, _, _ = self.episode([9], [28, 0, 0])
        self.assertEqual((summary["SR"], summary["OSR"], summary["stopped"]), (1, 1, False))
        self.assertEqual(summary["NE"], 19)
        self.assertEqual(summary["SPL"], 1)

    def test_passing_goal_then_overshooting_is_osr_only(self):
        summary, _, _ = self.episode([9] * 6, [30, 0, 0])
        self.assertEqual((summary["SR"], summary["OSR"], summary["SPL"]), (0, 1, 0))
        self.assertEqual(summary["NE"], 24)

    def test_stop_does_not_make_a_far_goal_successful(self):
        summary, records, _ = self.episode([0], [30, 0, 0])
        self.assertEqual((summary["SR"], summary["OSR"], summary["stopped"]), (0, 0, True))
        self.assertEqual([r["record_type"] for r in records], ["step", "sample_summary"])

    def test_history_limit_and_no_goal_input(self):
        _, _, base = self.episode([1] * 5, [100, 0, 0], history_images=2)
        self.assertEqual([len(call.args[1]) for call in base.policy.call_args_list], [1, 2, 2, 2, 2])
        self.assertTrue(all(len(call.args) == 3 and not call.kwargs for call in base.policy.call_args_list))

    def test_both_modes_resolve_same_ra50_dataset_and_camera(self):
        configs = []
        for mode in ("base", "spf"):
            selected, metadata = prepare_run(parse_args(["--mode", mode, "--dry-run"]))
            self.assertEqual(len(selected), 50)
            self.assertEqual(metadata["cameras"]["env_airsim_23"]["horizontal_fov_deg"], 90)
            metadata.pop("mode")
            configs.append(metadata)
        self.assertEqual(*configs)


if __name__ == "__main__":
    unittest.main()
