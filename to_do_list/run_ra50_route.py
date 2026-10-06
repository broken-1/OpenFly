"""Execute RA50 sample 0 with ordered Qwen subgoals and no MCTS."""

from dataclasses import asdict
import json
import os
from pathlib import Path

import cv2
import numpy as np

from eval import AirsimBridge, calculate_distance, getPoseAfterMakeAction, kill_env_process
from instruction_progress import (
    InstructionProgressController,
    build_action_prompt,
    build_tracker_prompt,
    parse_tracker_update,
)
from qwen_json_generator import (
    ACTION_NAMES,
    MOVEMENT_ACTIONS,
    QwenGeneratorConfig,
    QwenJsonGenerator,
)


def append_jsonl(path: Path, payload: dict):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def set_camera_pose(env_bridge, pose, pitch: float):
    env_bridge.set_camera_pose(
        pose[0],
        pose[1],
        pose[2],
        pitch,
        np.rad2deg(pose[3]),
        0,
    )


def main():
    eval_info = Path(
        os.environ.get(
            "OPENFLY_EVAL_INFO",
            "configs/seen_airsim_23_random50_seed20260609.json",
        )
    )
    results_path = Path(
        os.environ.get(
            "OPENFLY_TODO_ROUTE_RESULTS",
            "to_do_list/outputs/ra50_sample0_route.jsonl",
        )
    )
    image_dir = Path(
        os.environ.get(
            "OPENFLY_TODO_ROUTE_IMAGES",
            "to_do_list/outputs/ra50_sample0_route_images",
        )
    )
    max_step = int(os.environ.get("OPENFLY_TODO_MAX_STEP", "40"))
    history_images = int(os.environ.get("OPENFLY_TODO_HISTORY_IMAGES", "3"))
    with eval_info.open(encoding="utf-8") as handle:
        item = json.load(handle)[0]

    instruction = item["gpt_instruction"]
    positions = item["pos"]
    start = positions[0]
    goal = positions[-1]
    pose = [start[0], start[1], start[2], item["yaw"][0]]
    old_pose = list(pose)
    pitch = -45.0 if "high" in item["image_path"] else 0.0
    env_name = item["image_path"].split("/")[0]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    results_path.write_text("", encoding="utf-8")

    generator = QwenJsonGenerator(
        QwenGeneratorConfig(
            model_path=os.environ.get(
                "OPENFLY_QWEN_MODEL_PATH",
                "/home/liusongbo/models/Qwen3-VL-8B-Instruct",
            ),
            device=os.environ.get("OPENFLY_QWEN_DEVICE", "cuda:0"),
            attention=os.environ.get(
                "OPENFLY_QWEN_ATTN_IMPLEMENTATION", "flash_attention_2"
            ),
            max_new_tokens=int(os.environ.get("OPENFLY_TODO_MAX_NEW_TOKENS", "512")),
        )
    )
    controller = InstructionProgressController(generator)
    state = controller.create_plan(instruction)
    print(json.dumps(state.plan.to_dict(), ensure_ascii=False, indent=2))
    append_jsonl(
        results_path,
        {
            "record_type": "plan",
            "sample_idx": 0,
            "env_name": env_name,
            "instruction": instruction,
            "plan": state.plan.to_dict(),
        },
    )

    env_bridge = None
    action_history = []
    image_history = []
    reached_goal = False
    min_distance = calculate_distance(goal, pose)
    try:
        env_bridge = AirsimBridge(env_name)
        env_bridge.pass_len = 1e-3
        set_camera_pose(env_bridge, pose, pitch)

        for step in range(max_step):
            set_camera_pose(env_bridge, pose, pitch)
            current_image = env_bridge.get_camera_data()
            image_history.append(current_image)
            image_history = image_history[-history_images:]
            cv2.imwrite(str(image_dir / f"step_{step:03d}.jpg"), current_image)

            recent_action_names = [
                ACTION_NAMES[action] for action in action_history[-10:]
            ]
            tracker_prompt = build_tracker_prompt(
                state,
                step=step,
                recent_actions=recent_action_names,
            )
            raw_tracker = generator(tracker_prompt, image_history)
            tracker_update = parse_tracker_update(raw_tracker)
            transition = state.apply(tracker_update, step=step)

            action_decision = None
            if state.finished:
                selected_action = 0
            else:
                action_space = {
                    action: ACTION_NAMES[action] for action in MOVEMENT_ACTIONS
                }
                action_prompt = build_action_prompt(
                    state,
                    recent_actions=recent_action_names,
                    action_space=action_space,
                )
                action_decision = generator.choose_action(
                    action_prompt,
                    image_history,
                )
                selected_action = action_decision.action

            action_history.append(selected_action)
            current_distance = calculate_distance(goal, pose)
            min_distance = min(min_distance, current_distance)
            reached_goal = reached_goal or current_distance < 20
            print(
                f"[Step {step}] subgoal={tracker_update.subgoal_id} "
                f"status={tracker_update.status.value} "
                f"progress={tracker_update.progress:.2f} "
                f"action={ACTION_NAMES[selected_action]} "
                f"distance={current_distance:.2f}"
            )
            append_jsonl(
                results_path,
                {
                    "record_type": "step",
                    "sample_idx": 0,
                    "step": step,
                    "pose": pose,
                    "distance_to_goal": current_distance,
                    "tracker_update": asdict(tracker_update),
                    "transition": asdict(transition),
                    "active_subgoal": (
                        None if state.finished else asdict(state.active_subgoal)
                    ),
                    "action": selected_action,
                    "action_name": ACTION_NAMES[selected_action],
                    "action_decision": (
                        action_decision.to_dict() if action_decision else None
                    ),
                },
            )

            if selected_action == 0:
                break
            pose = getPoseAfterMakeAction(pose, selected_action)
            set_camera_pose(env_bridge, pose, pitch)
            env_bridge.pass_len += calculate_distance(old_pose, pose)
            old_pose = list(pose)

        final_distance = calculate_distance(goal, pose)
        stopped = bool(action_history and action_history[-1] == 0)
        success = int(stopped and final_distance < 20)
        shortest_path = max(calculate_distance(goal, start), 1e-12)
        actual_path = max(env_bridge.pass_len, shortest_path)
        summary = {
            "record_type": "sample_summary",
            "sample_idx": 0,
            "steps": len(action_history),
            "stopped": stopped,
            "plan_finished": state.finished,
            "final_pose": pose,
            "goal": goal,
            "NE": final_distance,
            "minimum_distance": min_distance,
            "SR": success,
            "OSR": int(reached_goal or success),
            "SPL": shortest_path / actual_path if success else 0.0,
            "actions": action_history,
            "action_names": [ACTION_NAMES[action] for action in action_history],
            "progress_state": state.to_dict(),
        }
        append_jsonl(results_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if env_bridge is not None:
            del env_bridge
        kill_env_process("AirVLN")

    print(f"saved: {results_path}")


if __name__ == "__main__":
    main()
