"""Evaluate Qwen3-VL with local visual MCTS in OpenFly environments."""

import atexit
from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from eval import (
    AirsimBridge,
    GSBridge,
    UEBridge,
    calculate_distance,
    getPoseAfterMakeAction,
    kill_env_process,
)
from mcts_search import MCTSSearch, SearchConfig
from qwen_vln import (
    ACTION_NAMES,
    TURN_ACTIONS,
    QwenConfig,
    QwenVLNBackend,
    TurnKeyframe,
)

MAX_TURN_KEYFRAMES = 3


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class EvalConfig:
    eval_info: str
    results_path: Path
    model: QwenConfig
    search: SearchConfig
    max_samples: int
    max_step: int
    start_sample: int
    keep_observations: bool
    observation_root: Path

    @classmethod
    def from_env(cls):
        keep_observations = env_bool("OPENFLY_MCTS_KEEP_OBS")
        default_obs_root = (
            "test/mcts_obs"
            if keep_observations
            else f"/tmp/openfly_qwen_mcts_obs_{os.getpid()}"
        )
        return cls(
            eval_info=os.environ.get(
                "OPENFLY_EVAL_INFO",
                "configs/seen_airsim_23_random50_seed20260609.json",
            ),
            results_path=Path(
                os.environ.get(
                    "OPENFLY_MCTS_RESULTS",
                    "outputs/mcts_eval/qwen3vl8b_mcts.jsonl",
                )
            ),
            model=QwenConfig(
                model_path=os.environ.get(
                    "OPENFLY_QWEN_MODEL_PATH",
                    "/home/liusongbo/models/Qwen3-VL-8B-Instruct",
                ),
                device=os.environ.get("OPENFLY_QWEN_DEVICE", "cuda:0"),
                attention=os.environ.get(
                    "OPENFLY_QWEN_ATTN_IMPLEMENTATION", "flash_attention_2"
                ),
            ),
            search=SearchConfig(
                top_k=int(os.environ.get("OPENFLY_MCTS_TOP_K", "3")),
                depth=int(os.environ.get("OPENFLY_MCTS_SEARCH_DEPTH", "2")),
                simulations=int(
                    os.environ.get("OPENFLY_MCTS_NUM_SIMULATIONS", "12")
                ),
                puct_c=float(os.environ.get("OPENFLY_MCTS_PUCT_C", "1.2")),
            ),
            max_samples=int(os.environ.get("OPENFLY_MCTS_MAX_SAMPLES", "0")),
            max_step=int(os.environ.get("OPENFLY_MCTS_MAX_STEP", "40")),
            start_sample=int(os.environ.get("OPENFLY_MCTS_START_SAMPLE", "0")),
            keep_observations=keep_observations,
            observation_root=Path(
                os.environ.get("OPENFLY_MCTS_OBS_DIR", default_obs_root)
            ),
        )


class PoseObserver:
    """Capture virtual-node observations and restore the real root pose."""

    def __init__(
        self,
        env_bridge,
        root_pose,
        pos_ratio: float,
        pitch: float,
        save_dir: Path | None,
    ):
        self.env_bridge = env_bridge
        self.root_pose = list(root_pose)
        self.pos_ratio = pos_ratio
        self.pitch = pitch
        self.save_dir = save_dir
        self.capture_count = 0

    def __call__(self, pose):
        set_camera_pose(
            self.env_bridge,
            pose,
            self.pos_ratio,
            self.pitch,
        )
        time.sleep(0.05)
        image = self.env_bridge.get_camera_data()
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(self.save_dir / f"virtual_{self.capture_count:03d}.png"),
                image,
            )
        self.capture_count += 1
        set_camera_pose(
            self.env_bridge,
            self.root_pose,
            self.pos_ratio,
            self.pitch,
        )
        return image


def set_camera_pose(env_bridge, pose, pos_ratio: float, pitch: float):
    env_bridge.set_camera_pose(
        pose[0] / pos_ratio,
        pose[1] / pos_ratio,
        pose[2] / pos_ratio,
        pitch,
        np.rad2deg(pose[3]),
        0,
    )


def create_env_bridge(env_name: str):
    if "airsim" in env_name:
        return AirsimBridge(env_name), 1.0
    if "ue" in env_name:
        return UEBridge("127.0.0.1", "9000", env_name), 1.0
    if "gs" in env_name:
        return GSBridge(env_name), 5.15
    raise ValueError(f"Unknown environment type: {env_name}")


def cleanup_env():
    for keyword in ("AirVLN", "guangzhou", "shanghai", "CitySample", "CrashReport"):
        kill_env_process(keyword)


def append_jsonl(path: Path, payload: dict):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_eval_groups(config: EvalConfig):
    with open(config.eval_info, "r", encoding="utf-8") as handle:
        all_items = json.load(handle)
    indexed_items = list(enumerate(all_items))[config.start_sample :]
    if config.max_samples > 0:
        indexed_items = indexed_items[: config.max_samples]

    groups = {}
    for sample_idx, item in indexed_items:
        env_name = item["image_path"].split("/")[0]
        groups.setdefault(env_name, []).append((sample_idx, item))
    return groups


def save_observation(path: Path | None, name: str, image):
    if path is None:
        return
    path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path / name), image)


def build_real_image_history(start_image, current_image, turn_keyframes, turn_count):
    """Return initial view, up to three turn views, and the current view."""
    images = []
    if turn_count <= MAX_TURN_KEYFRAMES:
        images.append(start_image)
    images.extend(keyframe.image for keyframe in turn_keyframes[-MAX_TURN_KEYFRAMES:])
    images.append(current_image)
    return images


def print_final_metrics(scene_metrics: dict, total: int, stop_total: int):
    print("\nQwen MCTS evaluation complete!")
    print(f"{'Scene':<20} {'NE/m':>10} {'SR/%':>10} {'OSR/%':>10} {'SPL/%':>10}")
    success_total = 0
    for env_name, metrics in scene_metrics.items():
        count = metrics["count"]
        success_total += metrics["sr"]
        print(
            f"{env_name:<20} {metrics['ne'] / count:>10.2f} "
            f"{metrics['sr'] / count * 100:>10.2f} "
            f"{metrics['osr'] / count * 100:>10.2f} "
            f"{metrics['spl'] / count * 100:>10.2f}"
        )
    print(f"Total samples: {total}")
    print(f"Final accuracy: {success_total / total if total else 0:.4f}")
    print(f"Final stop rate: {stop_total / total if total else 0:.4f}")


def evaluate_sample(
    env_name,
    sample_idx,
    item,
    env_bridge,
    pos_ratio,
    backend,
    search,
    config,
):
    positions = item["pos"]
    instruction = item["gpt_instruction"]
    start = positions[0]
    goal = positions[-1]
    pose = [start[0], start[1], start[2], item["yaw"][0]]
    old_pose = list(pose)
    pitch = -45.0 if "high" in item["image_path"] else 0.0
    action_history = []
    turn_keyframes = []
    turn_count = 0
    reached_goal = False
    env_bridge.pass_len = 1e-3
    sample_obs_dir = (
        config.observation_root / env_name / f"sample_{sample_idx}"
        if config.keep_observations
        else None
    )

    print(
        f"Sample {sample_idx}: {start} -> {goal}, "
        f"initial heading={item['yaw'][0]}"
    )
    set_camera_pose(env_bridge, pose, pos_ratio, pitch)
    start_image = env_bridge.get_camera_data()
    save_observation(sample_obs_dir, "start.png", start_image)

    for step in range(config.max_step):
        root_pose = list(pose)
        set_camera_pose(env_bridge, root_pose, pos_ratio, pitch)
        current_image = env_bridge.get_camera_data()
        cv2.imwrite("test/cur_img.jpg", current_image)
        real_image_history = build_real_image_history(
            start_image, current_image, turn_keyframes, turn_count
        )
        step_obs_dir = (
            sample_obs_dir / f"step_{step}" if sample_obs_dir is not None else None
        )
        save_observation(step_obs_dir, "current.png", current_image)
        observer = PoseObserver(
            env_bridge,
            root_pose,
            pos_ratio,
            pitch,
            step_obs_dir,
        )

        policy_output = backend.policy(
            instruction,
            real_image_history,
            action_history,
            allow_stop=True,
        )
        result = None
        if policy_output.action == 0:
            selected_action = 0
        else:
            result = search.search(
                instruction=instruction,
                root_pose=root_pose,
                root_policy_output=policy_output,
                real_image_history=real_image_history,
                action_history=action_history,
                observe=observer,
            )
            selected_action = result.selected_action

        previous_action = action_history[-1] if action_history else None
        if previous_action in TURN_ACTIONS:
            turn_count += 1
            turn_keyframes.append(
                TurnKeyframe(
                    step=step - 1,
                    action=previous_action,
                    image=current_image,
                )
            )
            turn_keyframes = turn_keyframes[-MAX_TURN_KEYFRAMES:]
        action_history.append(selected_action)

        if result is None:
            print(
                f"[Sample {sample_idx} Step {step}] "
                f"base=STOP execute=STOP"
            )
        else:
            print(
                f"[Sample {sample_idx} Step {step}] "
                f"base={ACTION_NAMES[result.base_action]}, "
                f"mcts={ACTION_NAMES[result.selected_action]}, "
                f"execute={ACTION_NAMES[selected_action]}, "
                f"search_depth={result.max_depth_reached}"
            )

        append_jsonl(
            config.results_path,
            {
                "record_type": "step",
                "env_name": env_name,
                "sample_idx": sample_idx,
                "step": step,
                "instruction": instruction,
                "root_pose": root_pose,
                "policy": policy_output.to_dict(),
                "base_action": policy_output.action,
                "base_action_name": ACTION_NAMES[policy_output.action],
                "candidates": result.candidates if result else [],
                "turn_keyframes": [
                    keyframe.metadata() for keyframe in turn_keyframes
                ],
                "simulations": result.simulations if result else [],
                "root_action_stats": result.root_action_stats if result else [],
                "max_depth_reached": result.max_depth_reached if result else 0,
                "mcts_action": result.selected_action if result else 0,
                "mcts_action_name": (
                    ACTION_NAMES[result.selected_action] if result else "STOP"
                ),
                "selected_action": selected_action,
                "selected_action_name": ACTION_NAMES[selected_action],
            },
        )

        set_camera_pose(env_bridge, root_pose, pos_ratio, pitch)
        if selected_action == 0:
            break

        pose = getPoseAfterMakeAction(pose, selected_action)
        set_camera_pose(env_bridge, pose, pos_ratio, pitch)
        env_bridge.pass_len += calculate_distance(old_pose, pose)
        if calculate_distance(goal, pose) < 20:
            reached_goal = True
        old_pose = list(pose)

    distance = calculate_distance(goal, pose)
    stopped = bool(action_history and action_history[-1] == 0)
    success = int(distance < 20)
    osr = int(reached_goal or success)
    shortest_path = max(calculate_distance(goal, start), 1e-12)
    actual_path = max(env_bridge.pass_len, shortest_path)
    spl = shortest_path / actual_path if success else 0.0
    env_bridge.distance_to_goal.append(distance)
    env_bridge.success.append(success)
    env_bridge.osr.append(osr)
    env_bridge.spl.append(spl)
    env_bridge.print_info()

    append_jsonl(
        config.results_path,
        {
            "record_type": "sample_summary",
            "env_name": env_name,
            "sample_idx": sample_idx,
            "steps": len(action_history),
            "stopped": stopped,
            "final_pose": pose,
            "goal": goal,
            "NE": distance,
            "SR": success,
            "OSR": osr,
            "SPL": spl,
            "actions": action_history,
            "action_names": [ACTION_NAMES[action] for action in action_history],
        },
    )
    return {
        "ne": distance,
        "sr": success,
        "osr": osr,
        "spl": spl,
        "stopped": stopped,
    }


def main():
    config = EvalConfig.from_env()
    config.results_path.parent.mkdir(parents=True, exist_ok=True)
    Path("test").mkdir(exist_ok=True)
    config.results_path.write_text("", encoding="utf-8")
    config.observation_root.mkdir(parents=True, exist_ok=True)
    if not config.keep_observations:
        atexit.register(
            lambda: shutil.rmtree(config.observation_root, ignore_errors=True)
        )

    print(
        "Qwen MCTS: "
        f"model={config.model.model_path}, top_k={config.search.top_k}, "
        f"depth={config.search.depth}, simulations={config.search.simulations}, "
        f"puct_c={config.search.puct_c}, results={config.results_path}"
    )
    backend = QwenVLNBackend(config.model)
    search = MCTSSearch(backend, getPoseAfterMakeAction, config.search)

    total = 0
    stop_total = 0
    scene_metrics = {}
    for env_name, items in load_eval_groups(config).items():
        print(f"Starting Qwen MCTS: {env_name}, samples={len(items)}")
        time.sleep(5)
        env_bridge, pos_ratio = create_env_bridge(env_name)
        try:
            metrics = {"count": 0, "ne": 0.0, "sr": 0, "osr": 0, "spl": 0.0}
            scene_metrics[env_name] = metrics
            for sample_idx, item in items:
                result = evaluate_sample(
                    env_name,
                    sample_idx,
                    item,
                    env_bridge,
                    pos_ratio,
                    backend,
                    search,
                    config,
                )
                total += 1
                stop_total += result["stopped"]
                metrics["count"] += 1
                for name in ("ne", "sr", "osr", "spl"):
                    metrics[name] += result[name]
        finally:
            cleanup_env()
            del env_bridge
            gc.collect()

    print_final_metrics(scene_metrics, total, stop_total)
    print(f"Qwen MCTS results log: {config.results_path}")


if __name__ == "__main__":
    main()
