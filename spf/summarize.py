"""Verify a completed evaluation before presenting RA50 aggregate results."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean


def summarize(path):
    with Path(path).open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    configs = [r for r in records if r["record_type"] == "run_config"]
    finals = [r for r in records if r["record_type"] == "run_summary"]
    samples = [r for r in records if r["record_type"] == "sample_summary"]
    if len(configs) != 1 or len(finals) != 1 or records[-1] != finals[0]:
        raise ValueError("Expected one run_config and a terminal run_summary; run may be incomplete")
    config, final = configs[0], finals[0]
    expected = config["sample_indices"]
    actual = [r["sample_idx"] for r in samples]
    if len(actual) != len(set(actual)) or sorted(actual) != sorted(expected):
        raise ValueError("Completed sample indices do not exactly match the requested set")
    if final["samples"] != len(samples) or final["mode"] != config["mode"]:
        raise ValueError("Terminal summary disagrees with run configuration")
    steps = defaultdict(list)
    for record in records:
        if record["record_type"] == "step":
            steps[record["sample_idx"]].append(record)
    if set(steps) != set(expected):
        raise ValueError("Missing steps or steps outside the requested sample set")
    for sample in samples:
        trace = steps[sample["sample_idx"]]
        if [s["step"] for s in trace] != list(range(sample["steps"])):
            raise ValueError(f"Non-contiguous steps for sample {sample['sample_idx']}")
        if [s["action"] for s in trace] != sample["actions"]:
            raise ValueError(f"Action trace disagrees with sample {sample['sample_idx']}")
        if not 1 <= sample["steps"] <= config["max_step"]:
            raise ValueError("Step count is outside configured bounds")
        if 0 in sample["actions"][:-1]:
            raise ValueError("Trajectory continued after STOP")
        if sample["stopped"] != (sample["actions"][-1] == 0):
            raise ValueError("STOP flag disagrees with trajectory")
        ne = math.dist(sample["final_pose"][:3], sample["goal"][:3])
        if not math.isclose(ne, sample["NE"], abs_tol=1e-8) or sample["SR"] != int(ne < 20):
            raise ValueError("Reported terminal navigation metric disagrees with recorded pose")
        poses = [s["pose"][:3] for s in trace] + [sample["final_pose"][:3]]
        goal = sample["goal"][:3]
        osr = int(any(math.dist(pose, goal) < 20 for pose in poses[1:]) or sample["SR"])
        path_length = 1e-3 + sum(math.dist(a, b) for a, b in zip(poses, poses[1:]))
        shortest = max(math.dist(poses[0], goal), 1e-12)
        spl = shortest / max(path_length, shortest) if sample["SR"] else 0.0
        if sample["OSR"] != osr or not math.isclose(sample["SPL"], spl, abs_tol=1e-8):
            raise ValueError("Reported OSR/SPL disagrees with recorded trajectory")
    for key in ("NE", "SR", "OSR", "SPL"):
        if not math.isclose(mean(s[key] for s in samples), final[key], abs_tol=1e-9):
            raise ValueError(f"Aggregate {key} disagrees with sample summaries")
    all_steps = [s for trace in steps.values() for s in trace]
    sources = Counter(s["decision"]["source"] for s in all_steps)
    return {
        "artifact": str(Path(path).resolve()), "verified_complete": True,
        "mode": config["mode"], "samples": len(samples),
        "dataset_sha256": config["dataset_sha256"],
        "max_step": config["max_step"], "history_images": config["history_images"],
        "NE_m": final["NE"],
        **{f"{key}_percent": final[key] * 100 for key in ("SR", "OSR", "SPL")},
        "success_count": sum(s["SR"] for s in samples),
        "success_indices": [s["sample_idx"] for s in samples if s["SR"]],
        "oracle_success_count": sum(s["OSR"] for s in samples),
        "stop_count": sum(s["stopped"] for s in samples),
        "total_steps": len(all_steps), "mean_steps": mean(s["steps"] for s in samples),
        "decision_sources": dict(sources),
        "changed_from_baseline_steps": sum(
            s["action"] != s["decision"]["baseline"]["action"] for s in all_steps),
        "mean_decision_seconds": mean(s["decision_seconds"] for s in all_steps),
        "episode_seconds": sum(s["elapsed_seconds"] for s in samples),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
