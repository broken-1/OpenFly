import json
import os
import time
from pathlib import Path

import numpy as np
import psutil

from eval import (
    AirsimBridge,
    GSBridge,
    UEBridge,
    calculate_distance,
    getPoseAfterMakeAction,
)
from qwen_vln import ACTION_NAMES, QwenConfig, QwenVLNBackend

MODEL_PATH = os.environ.get(
    "OPENFLY_QWEN_MODEL_PATH",
    "/home/liusongbo/models/Qwen3-VL-8B-Instruct",
)
EVAL_INFO = os.environ.get(
    "OPENFLY_EVAL_INFO",
    "configs/seen_airsim_23_random50_seed20260609.json",
)
RESULTS_PATH = Path(
    os.environ.get("OPENFLY_QWEN_RESULTS", "outputs/qwen_eval_results.jsonl")
)
MAX_SAMPLES = int(os.environ.get("OPENFLY_QWEN_MAX_SAMPLES", "5"))
MAX_STEP = int(os.environ.get("OPENFLY_QWEN_MAX_STEP", "40"))
HISTORY_IMAGES = int(os.environ.get("OPENFLY_QWEN_HISTORY_IMAGES", "3"))
DEVICE = os.environ.get("OPENFLY_QWEN_DEVICE", "cuda:0")
ATTN_IMPLEMENTATION = os.environ.get(
    "OPENFLY_QWEN_ATTN_IMPLEMENTATION", "flash_attention_2"
)
START_SAMPLE = int(os.environ.get("OPENFLY_QWEN_START_SAMPLE", "0"))
APPEND_RESULTS = os.environ.get("OPENFLY_QWEN_APPEND_RESULTS", "0").lower() in {
    "1",
    "true",
    "yes",
}


def create_env_bridge(env_name):
    if "airsim" in env_name:
        return OwnedAirsimBridge(env_name), 1.0
    if "ue" in env_name:
        return OwnedUEBridge("127.0.0.1", "9000", env_name), 1.0
    if "gs" in env_name:
        return GSBridge(env_name), 5.15
    raise ValueError(f"Unknown environment type: {env_name}")


class OwnedAirsimBridge(AirsimBridge):
    def _init_airsim_sim(self):
        import subprocess
        import shlex
        command = ["bash", f"envs/airsim/{self.env_name}/LinuxNoEditor/start.sh"]
        command += shlex.split(os.environ.get("OPENFLY_AIRSIM_EXTRA_ARGS", ""))
        sim_env = os.environ.copy()
        sim_env["CUDA_VISIBLE_DEVICES"] = os.environ.get(
            "OPENFLY_AIRSIM_CUDA_VISIBLE_DEVICES", sim_env.get("CUDA_VISIBLE_DEVICES", "1")
        )
        with RESULTS_PATH.with_suffix(".simulator.log").open("a") as log:
            self.process = subprocess.Popen(command, env=sim_env, stdout=log,
                                            stderr=subprocess.STDOUT, text=True)
            code = self.process.wait()
        if code:
            print(f"Simulator {self.env_name} exited with code {code}", flush=True)


class OwnedUEBridge(UEBridge):
    def kill_failed_process(self):
        # The evaluator owns its simulator; do not kill another user's CitySample.
        pass

    def _init_ue_sim(self):
        import subprocess
        import shlex
        command = ["bash", f"envs/ue/{self.env_name}/CitySample.sh"]
        command += shlex.split(os.environ.get("OPENFLY_UE_EXTRA_ARGS", ""))
        sim_env = os.environ.copy()
        sim_env["CUDA_VISIBLE_DEVICES"] = os.environ.get(
            "OPENFLY_AIRSIM_CUDA_VISIBLE_DEVICES", sim_env.get("CUDA_VISIBLE_DEVICES", "1")
        )
        with RESULTS_PATH.with_suffix(".simulator.log").open("a") as log:
            self.process = subprocess.Popen(command, env=sim_env, stdout=log,
                                            stderr=subprocess.STDOUT, text=True)
            self.process.wait()


def cleanup_env():
    # Only terminate descendants of this evaluator, never global name matches.
    processes = psutil.Process().children(recursive=True)
    for process in reversed(processes):
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=5)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass


def append_jsonl(payload):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def prepare_results_file():
    """Prepare a JSONL checkpoint and return summaries retained from an earlier run."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not APPEND_RESULTS or not RESULTS_PATH.exists():
        RESULTS_PATH.write_text("", encoding="utf-8")
        return []

    retained = []
    summaries = []
    with RESULTS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_idx = record.get("sample_idx")
            # A prior interruption can leave partial step records for the next
            # sample. Drop them so resuming does not duplicate that trajectory.
            if sample_idx is not None and sample_idx >= START_SAMPLE:
                continue
            retained.append(record)
            if record.get("record_type") == "sample_summary":
                summaries.append(record)

    completed = [record["sample_idx"] for record in summaries]
    if sorted(completed) != list(range(START_SAMPLE)):
        raise ValueError(
            "Resume checkpoint must contain exactly one completed summary for "
            f"every sample before {START_SAMPLE}; original file was not changed"
        )

    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        for record in retained:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return summaries


def seed_metrics_from_summaries(summaries):
    scene_metrics = {}
    for summary in summaries:
        env_name = summary["env_name"]
        metrics = scene_metrics.setdefault(
            env_name,
            {"count": 0, "ne": 0.0, "sr": 0.0, "osr": 0.0, "spl": 0.0},
        )
        metrics["count"] += 1
        metrics["ne"] += float(summary["NE"])
        metrics["sr"] += int(summary["SR"])
        metrics["osr"] += int(summary["OSR"])
        metrics["spl"] += float(summary["SPL"])
    return scene_metrics


def main():
    if MAX_STEP <= 0 or HISTORY_IMAGES <= 0:
        raise ValueError("MAX_STEP and HISTORY_IMAGES must be positive")
    with open(EVAL_INFO, "r", encoding="utf-8") as handle:
        all_eval_info = json.load(handle)
    if START_SAMPLE < 0 or START_SAMPLE > len(all_eval_info):
        raise ValueError(
            f"OPENFLY_QWEN_START_SAMPLE must be between 0 and {len(all_eval_info)}"
        )
    indexed_eval_info = list(enumerate(all_eval_info))[START_SAMPLE:]
    if MAX_SAMPLES > 0:
        indexed_eval_info = indexed_eval_info[:MAX_SAMPLES]

    previous_summaries = prepare_results_file()
    print(
        f"Qwen baseline: model={MODEL_PATH}, samples={len(indexed_eval_info)}, "
        f"start_sample={START_SAMPLE}, append_results={APPEND_RESULTS}, "
        f"max_step={MAX_STEP}, history_images={HISTORY_IMAGES}, "
        f"attention={ATTN_IMPLEMENTATION}, results={RESULTS_PATH}"
    )
    policy_config = QwenConfig(
        model_path=MODEL_PATH,
        device=DEVICE,
        attention=ATTN_IMPLEMENTATION,
        history_images=HISTORY_IMAGES,
    )
    if os.environ.get("OPENFLY_QWEN_BACKEND", "transformers") == "vllm":
        from qwen_vllm import VllmQwenVLNBackend
        policy = VllmQwenVLNBackend(
            policy_config,
            os.environ.get("OPENFLY_QWEN_API_URL", "http://127.0.0.1:8004/v1"),
            os.environ.get("OPENFLY_QWEN_API_MODEL", "qwen3.8-27b-fp8"),
        )
    else:
        policy = QwenVLNBackend(policy_config)

    env_groups = {}
    for sample_idx, item in indexed_eval_info:
        env_groups.setdefault(item["image_path"].split("/")[0], []).append(
            (sample_idx, item)
        )

    total = len(previous_summaries)
    success_total = sum(int(summary["SR"]) for summary in previous_summaries)
    stop_total = sum(bool(summary.get("stopped")) for summary in previous_summaries)
    scene_metrics = seed_metrics_from_summaries(previous_summaries)

    for env_name, items in env_groups.items():
        print(f"Starting Qwen evaluation: {env_name}, samples={len(items)}")
        time.sleep(5)
        try:
            env_bridge, pos_ratio = create_env_bridge(env_name)
            for sample_idx, item in items:
                total += 1
                positions = item["pos"]
                instruction = item["gpt_instruction"]
                start = positions[0]
                goal = positions[-1]
                pose = [start[0], start[1], start[2], item["yaw"][0]]
                old_pose = list(pose)
                pitch = -45.0 if "high" in item["image_path"] else 0.0
                actions = []
                image_history = []
                flag_osr = calculate_distance(goal, start) < 20
                env_bridge.pass_len = 1e-3

                print(
                    f"Sample {sample_idx}: {start} -> {goal}, "
                    f"initial heading={item['yaw'][0]}"
                )
                env_bridge.set_camera_pose(
                    pose[0] / pos_ratio,
                    pose[1] / pos_ratio,
                    pose[2] / pos_ratio,
                    pitch,
                    np.rad2deg(pose[3]),
                    0,
                )

                for step in range(MAX_STEP):
                    raw_image = env_bridge.get_camera_data()
                    if raw_image is None or not raw_image.size:
                        raise RuntimeError("Simulator returned an empty camera image")
                    # UnrealCV decodes with OpenCV (BGR); the policy expects RGB.
                    if "ue" in env_name:
                        raw_image = raw_image[..., ::-1].copy()
                    image_history.append(raw_image)
                    image_history = image_history[-HISTORY_IMAGES:]
                    policy_output = policy.policy(
                        instruction,
                        image_history[-policy.config.history_images :],
                        actions,
                    )
                    action = policy_output.action
                    action_probability = policy_output.probabilities[action]
                    actions.append(action)
                    print(
                        f"[Sample {sample_idx} Step {step}] "
                        f"{ACTION_NAMES[action]} ({action}), "
                        f"p={action_probability:.4f}"
                    )
                    append_jsonl(
                        {
                            "record_type": "step",
                            "env_name": env_name,
                            "sample_idx": sample_idx,
                            "step": step,
                            "instruction": instruction,
                            "pose": pose,
                            "action": action,
                            "action_name": ACTION_NAMES[action],
                            "action_probability": action_probability,
                            "policy": policy_output.to_dict(),
                        }
                    )

                    if action == 0:
                        stop_total += 1
                        break

                    pose = getPoseAfterMakeAction(pose, action)
                    env_bridge.set_camera_pose(
                        pose[0] / pos_ratio,
                        pose[1] / pos_ratio,
                        pose[2] / pos_ratio,
                        pitch,
                        np.rad2deg(pose[3]),
                        0,
                    )
                    env_bridge.pass_len += calculate_distance(old_pose, pose)
                    if calculate_distance(goal, pose) < 20:
                        flag_osr = True
                    old_pose = list(pose)

                distance = calculate_distance(goal, pose)
                success = int(distance < 20)
                success_total += success
                shortest_path = max(calculate_distance(goal, start), 1e-12)
                actual_path = max(env_bridge.pass_len, shortest_path)
                spl = shortest_path / actual_path if success else 0.0
                osr = int(flag_osr or success)

                env_bridge.distance_to_goal.append(distance)
                env_bridge.success.append(success)
                env_bridge.osr.append(osr)
                env_bridge.spl.append(spl)
                env_bridge.print_info()

                append_jsonl(
                    {
                        "record_type": "sample_summary",
                        "env_name": env_name,
                        "sample_idx": sample_idx,
                        "steps": len(actions),
                        "stopped": bool(actions and actions[-1] == 0),
                        "final_pose": pose,
                        "goal": goal,
                        "NE": distance,
                        "SR": success,
                        "OSR": osr,
                        "SPL": spl,
                        "actions": actions,
                        "action_names": [ACTION_NAMES[action] for action in actions],
                    }
                )

                metrics = scene_metrics.setdefault(
                    env_name,
                    {"count": 0, "ne": 0.0, "sr": 0.0, "osr": 0.0, "spl": 0.0},
                )
                metrics["count"] += 1
                metrics["ne"] += distance
                metrics["sr"] += success
                metrics["osr"] += osr
                metrics["spl"] += spl
        finally:
            cleanup_env()

    print("\nQwen evaluation complete!")
    print(f"{'Scene':<20} {'NE/m':>10} {'SR/%':>10} {'OSR/%':>10} {'SPL/%':>10}")
    for env_name, metrics in scene_metrics.items():
        count = metrics["count"]
        print(
            f"{env_name:<20} {metrics['ne'] / count:>10.2f} "
            f"{metrics['sr'] / count * 100:>10.2f} "
            f"{metrics['osr'] / count * 100:>10.2f} "
            f"{metrics['spl'] / count * 100:>10.2f}"
        )
    print(f"Total samples: {total}")
    print(f"Final accuracy: {success_total / total if total else 0:.4f}")
    print(f"Final stop rate: {stop_total / total if total else 0:.4f}")
    print(f"Qwen results log: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
