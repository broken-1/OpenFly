import unittest

from to_do_list.instruction_progress import (
    FINAL_TARGET_CRITERION,
    INTERMEDIATE_CRITERION_FALLBACK,
    InstructionProgressController,
    ProgressState,
    SubgoalStatus,
    TrackerStatus,
    TrackerUpdate,
    build_tracker_prompt,
    parse_instruction_plan,
)


class InstructionProgressTest(unittest.TestCase):
    def make_plan(self):
        return parse_instruction_plan(
            {
                "final_target": "the red tower entrance",
                "subgoals": [
                    {
                        "description": "pass the park",
                        "completion_criterion": "the park is behind the drone",
                    },
                    {
                        "description": "the red tower entrance",
                        "completion_criterion": "the entrance is visible",
                    },
                ],
            },
            "Pass the park and stop at the red tower entrance.",
        )

    def test_final_target_is_last_and_uses_20_meter_criterion(self):
        plan = self.make_plan()
        self.assertEqual(len(plan.subgoals), 2)
        self.assertTrue(plan.final_subgoal.is_final)
        self.assertEqual(
            plan.final_subgoal.completion_criterion,
            FINAL_TARGET_CRITERION,
        )

    def test_duplicate_final_arrival_is_normalized(self):
        plan = parse_instruction_plan(
            {
                "final_target": "gray building facade",
                "subgoals": [
                    {
                        "description": "turn left and continue to gray building facade",
                        "completion_criterion": "reach within 20 meters and stop",
                    },
                    {
                        "description": "gray building facade",
                        "completion_criterion": "stop at final target",
                    },
                ],
            },
            "Turn left and reach the gray building facade.",
        )
        self.assertEqual(
            [subgoal.description for subgoal in plan.subgoals],
            ["turn left and continue", "gray building facade"],
        )
        self.assertEqual(
            plan.subgoals[0].completion_criterion,
            INTERMEDIATE_CRITERION_FALLBACK,
        )

    def test_final_target_articles_do_not_create_duplicate_step(self):
        plan = parse_instruction_plan(
            {
                "final_target": "a modern building facade",
                "subgoals": [
                    {
                        "description": "Reach the modern building facade",
                        "completion_criterion": "reach within 20 meters and stop",
                    }
                ],
            },
            "Reach a modern building facade.",
        )
        self.assertEqual(len(plan.subgoals), 1)
        self.assertTrue(plan.final_subgoal.is_final)

    def test_low_confidence_completion_does_not_advance(self):
        state = ProgressState(self.make_plan())
        transition = state.apply(
            TrackerUpdate(1, TrackerStatus.COMPLETED, 1.0, 0.5, "weak evidence"),
            step=4,
        )
        self.assertFalse(transition.advanced)
        self.assertEqual(state.active_subgoal.id, 1)
        self.assertEqual(state.statuses[0], SubgoalStatus.ACTIVE)

    def test_completion_advances_exactly_one_subgoal(self):
        state = ProgressState(self.make_plan())
        transition = state.apply(
            TrackerUpdate(1, TrackerStatus.COMPLETED, 1.0, 0.9, "park behind"),
            step=4,
        )
        self.assertTrue(transition.advanced)
        self.assertFalse(transition.plan_finished)
        self.assertEqual(state.active_subgoal.id, 2)
        self.assertEqual(state.statuses[0], SubgoalStatus.COMPLETED)
        self.assertEqual(state.statuses[1], SubgoalStatus.ACTIVE)

    def test_tracker_prompt_changes_with_active_subgoal(self):
        state = ProgressState(self.make_plan())
        first_prompt = build_tracker_prompt(state, step=0, recent_actions=[])
        state.apply(
            TrackerUpdate(1, TrackerStatus.COMPLETED, 1.0, 0.9, "park behind"),
            step=1,
        )
        second_prompt = build_tracker_prompt(state, step=2, recent_actions=["FORWARD_3"])
        self.assertIn("subgoal 1 of 2", first_prompt)
        self.assertIn("subgoal 2 of 2", second_prompt)
        self.assertNotEqual(first_prompt, second_prompt)

    def test_controller_uses_planner_then_tracker(self):
        responses = iter(
            [
                (
                    '{"final_target":"tower","subgoals":['
                    '{"description":"tower","completion_criterion":"tower is near"}]}'
                ),
                (
                    '{"subgoal_id":1,"status":"COMPLETED",'
                    '"progress":1.0,"confidence":0.9,"evidence":"tower is near"}'
                ),
            ]
        )

        def generate(prompt, images):
            return next(responses)

        controller = InstructionProgressController(generate)
        state = controller.create_plan("Fly to the tower.")
        transition = controller.update(state, 3, images=["image"], recent_actions=[])
        self.assertTrue(transition.plan_finished)


if __name__ == "__main__":
    unittest.main()
