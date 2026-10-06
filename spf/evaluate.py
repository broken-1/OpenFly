"""Paired base/SPF AirSim evaluation, with a dependency-free configuration check.

Run from the project root with python -m spf.evaluate (or bash spf/run_eval.sh).
"""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from .policy import SpfPolicy
from .waypoint import CONTROLLER_VERSION, PROMPT_VERSION, CameraGeometry


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_DIR / "configs/seen_airsim_23_random50_seed20260609.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("base", "spf"), default="spf")
    parser.add_argument("--eval-info", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-path", type=Path, default=Path(os.environ.get(
        "OPENFLY_QWEN_MODEL_PATH", "/home/liusongbo/models/Qwen3-VL-8B-Instruct")))
    parser.add_argument("--max-samples", type=int, default=50, help="0 means all remaining samples")
    parser.add_argument("--start-sample", type=int, default=0)
    parser.add_argument("--max-step", type=int, default=40)
    parser.add_argument("--history-images", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--device", default=os.environ.get("OPENFLY_QWEN_DEVICE", "cuda:0"))
    parser.add_argument("--attention", default=os.environ.get(
        "OPENFLY_QWEN_ATTN_IMPLEMENTATION", "flash_attention_2"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results", type=Path, help="New JSONL file; existing files are never overwritten")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print configuration without loading Qwen/AirSim")
    parser.add_argument("--image", type=Path, help="Run one real-model decision on this image; no simulator or navigation metrics")
    args = parser.parse_args(argv)
    for name in ("max_step", "history_images", "max_new_tokens"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_samples < 0 or args.start_sample < 0:
        parser.error("--max-samples and --start-sample must be nonnegative")
    if args.image is not None and args.results is not None:
        parser.error("--image prints to stdout; --results is only for navigation evaluation")
    return args


def read_camera(env_name):
    path = PROJECT_DIR / "envs/airsim" / env_name / "LinuxNoEditor/AirVLN/Binaries/Linux/settings.json"
    with path.open(encoding="utf-8") as handle:
        settings = json.load(handle)
    cameras = [
        vehicle["Cameras"]["front_custom"]
        for vehicle in settings["Vehicles"].values()
        if "front_custom" in vehicle.get("Cameras", {})
    ]
    if len(cameras) != 1:
        raise ValueError(f"Expected one front_custom camera in {path}")
    camera = cameras[0]
    if camera.get("Yaw", 0) != 0 or camera.get("Roll", 0) != 0:
        raise ValueError("SPF v1 supports only a camera with zero mount yaw and roll")
    capture = next(item for item in camera["CaptureSettings"] if item["ImageType"] == 0)
    fov = capture["FOV_Degrees"]
    CameraGeometry(horizontal_fov_deg=fov)
    return {
        "settings_path": str(path),
        "settings_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "horizontal_fov_deg": fov,
        "mount_pitch_deg": float(camera.get("Pitch", 0)),
    }


def prepare_run(args):
    data_bytes = args.eval_info.read_bytes()
    items = json.loads(data_bytes)
    if not isinstance(items, list) or not 0 <= args.start_sample < len(items):
        raise ValueError("Dataset must be a nonempty list and start-sample must index it")
    selected = list(enumerate(items))[args.start_sample:]
    if args.max_samples:
        selected = selected[:args.max_samples]
    env_names = sorted({item["image_path"].split("/")[0] for _, item in selected})
    if any(not name.startswith("env_airsim_") or name in (".", "..") for name in env_names):
        raise ValueError("SPF v1 evaluates AirSim scenes only")
    cameras = {name: read_camera(name) for name in env_names}
    for index, item in selected:
        if not item.get("gpt_instruction") or not item.get("pos") or not item.get("yaw"):
            raise ValueError(f"Incomplete navigation sample: {index}")
    source_paths = [
        "mcts/qwen_vln.py", "train/eval.py", "spf/waypoint.py", "spf/policy.py",
        "spf/qwen_backend.py", "spf/evaluate.py",
    ]
    metadata = {
        "record_type": "run_config",
        "mode": args.mode,
        "eval_info": str(args.eval_info.resolve()),
        "dataset_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "sample_indices": [index for index, _ in selected],
        "model_path": str(args.model_path.resolve()),
        "max_step": args.max_step,
        "history_images": args.history_images,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "attention": args.attention,
        "seed": args.seed,
        "waypoint_prompt_version": PROMPT_VERSION,
        "controller_version": CONTROLLER_VERSION,
        "controller": asdict(CameraGeometry()),
        "cameras": cameras,
        "source_sha256": {
            path: hashlib.sha256((PROJECT_DIR / path).read_bytes()).hexdigest()
            for path in source_paths
        },
        "metrics": {
            "SR": "final NE < 20 m; STOP is not required, matching mcts/base_qwen.py",
            "OSR": "any post-action pose within 20 m, or final SR",
            "SPL": "SR * straight_line_start_goal / max(path_length, straight_line_start_goal)",
        },
    }
    return selected, metadata


def run_episode(bridge, policy, item, sample_idx, max_step, history_images,
                action_names, move_pose, distance, emit, mount_pitch_deg=0.0):
    """Same observations, action execution, and metric conventions for both arms.

    Only instruction, RGB history, executed actions, and camera calibration are
    passed to the policy. Ground-truth goal coordinates are used for scoring only.
    """
    env_name = item["image_path"].split("/")[0]
    start, goal = item["pos"][0], item["pos"][-1]
    pose = [*start[:3], item["yaw"][0]]
    pitch = -45.0 if "high" in item["image_path"] else 0.0
    actions, images = [], []
    path_length, reached = 1e-3, False
    sources = Counter()

    def set_pose():
        bridge.set_camera_pose(*pose[:3], pitch, math.degrees(pose[3]), 0)

    set_pose()
    started = time.perf_counter()
    for step in range(max_step):
        image = bridge.get_camera_data()
        images = (images + [image])[-history_images:]
        decision_started = time.perf_counter()
        decision = policy.policy(
            item["gpt_instruction"], images, list(actions),
            image_size=(image.shape[1], image.shape[0]),
            pitch_deg=pitch + mount_pitch_deg,
        )
        decision_seconds = time.perf_counter() - decision_started
        action = decision.action
        actions.append(action)
        sources[decision.source] += 1
        emit({
            "record_type": "step", "env_name": env_name, "sample_idx": sample_idx,
            "step": step, "pose": list(pose), "instruction": item["gpt_instruction"],
            "action": action, "action_name": action_names[action],
            "decision": decision.to_dict(), "decision_seconds": decision_seconds,
        })
        print(f"[Sample {sample_idx} Step {step}] {action_names[action]} ({decision.source})", flush=True)
        if action == 0:
            break
        old_pose = list(pose)
        pose = move_pose(pose, action)
        set_pose()
        path_length += distance(old_pose, pose)
        reached = reached or distance(goal, pose) < 20

    ne = distance(goal, pose)
    sr = int(ne < 20)
    shortest = max(distance(goal, start), 1e-12)
    summary = {
        "record_type": "sample_summary", "env_name": env_name, "sample_idx": sample_idx,
        "steps": len(actions), "stopped": bool(actions and actions[-1] == 0),
        "final_pose": pose, "goal": goal, "NE": ne, "SR": sr,
        "OSR": int(reached or sr), "SPL": shortest / max(path_length, shortest) if sr else 0.0,
        "actions": actions, "action_names": [action_names[a] for a in actions],
        "path_length": path_length, "decision_sources": dict(sources),
        "elapsed_seconds": time.perf_counter() - started,
    }
    emit(summary)
    return summary


def close_owned_environment(bridge):
    """Stop only the process tree started by this evaluation's bridge."""
    import psutil

    process = getattr(bridge, "process", None)
    if process is None or process.poll() is not None:
        return
    try:
        root = psutil.Process(process.pid)
        owned = root.children(recursive=True) + [root]
    except psutil.NoSuchProcess:
        return
    for child in owned:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(owned, timeout=5)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass


def main(argv=None):
    args = parse_args(argv)
    selected, metadata = prepare_run(args)
    if args.dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return
    if args.results is not None and args.results.exists():
        raise FileExistsError(f"Use a new results path: {args.results}")
    if args.image is not None and not args.image.is_file():
        raise FileNotFoundError(args.image)

    import numpy as np
    import torch
    import transformers
    from mcts.qwen_vln import ACTION_NAMES, POLICY_PROMPT_VERSION, QwenConfig, QwenVLNBackend
    from .qwen_backend import QwenWaypointGenerator

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver/device access; use --dry-run for CPU configuration checks.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    metadata["baseline_prompt_version"] = POLICY_PROMPT_VERSION
    metadata["runtime_versions"] = {"torch": torch.__version__, "transformers": transformers.__version__}
    backend = QwenVLNBackend(QwenConfig(
        str(args.model_path), args.device, args.attention, args.history_images,
    ))
    generator = QwenWaypointGenerator(backend, args.max_new_tokens)

    def make_policy(env_name):
        return SpfPolicy(
            backend, generator, ACTION_NAMES,
            CameraGeometry(horizontal_fov_deg=metadata["cameras"][env_name]["horizontal_fov_deg"]),
            enabled=args.mode == "spf",
        )

    if args.image is not None:
        from PIL import Image

        index, item = selected[0]
        env_name = item["image_path"].split("/")[0]
        with Image.open(args.image) as source:
            image = source.convert("RGB")
        pitch = -45.0 if "high" in item["image_path"] else 0.0
        decision = make_policy(env_name).policy(
            item["gpt_instruction"], [image], [], image_size=image.size,
            pitch_deg=pitch + metadata["cameras"][env_name]["mount_pitch_deg"],
        )
        print(json.dumps({
            "record_type": "image_smoke", "sample_idx": index,
            "image_path": str(args.image.resolve()), "image_is_verified_sample_frame": False,
            "config": metadata, "decision": decision.to_dict(),
        }, ensure_ascii=False, indent=2))
        return

    # Existing simulator imports require train/ on sys.path and root as cwd.
    sys.path.insert(0, str(PROJECT_DIR / "train"))
    from eval import AirsimBridge, calculate_distance, getPoseAfterMakeAction

    class EvaluationAirsimBridge(AirsimBridge):
        def __init__(self, env_name):
            try:
                super().__init__(env_name)
            except BaseException:
                close_owned_environment(self)
                raise

    os.chdir(PROJECT_DIR)
    results = args.results or PROJECT_DIR / "spf/outputs" / (
        f"{args.mode}_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}.jsonl")
    results.parent.mkdir(parents=True, exist_ok=True)
    groups = {}
    for index, item in selected:
        groups.setdefault(item["image_path"].split("/")[0], []).append((index, item))
    summaries = []
    with results.open("x", encoding="utf-8") as handle:
        def emit(payload):
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()

        emit(metadata)
        print(f"Results: {results}", flush=True)
        for env_name, items in groups.items():
            bridge = None
            try:
                bridge = EvaluationAirsimBridge(env_name)
                policy = make_policy(env_name)
                for index, item in items:
                    summaries.append(run_episode(
                        bridge, policy, item, index, args.max_step, args.history_images,
                        ACTION_NAMES, getPoseAfterMakeAction, calculate_distance, emit,
                        metadata["cameras"][env_name]["mount_pitch_deg"],
                    ))
            finally:
                if bridge is not None:
                    close_owned_environment(bridge)
        aggregate = {
            "record_type": "run_summary", "mode": args.mode, "samples": len(summaries),
            **{metric: sum(s[metric] for s in summaries) / len(summaries)
               for metric in ("NE", "SR", "OSR", "SPL")},
        }
        emit(aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
