"""Record committed source and explicit configuration before an evaluation starts."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()

    status = git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise SystemExit("Commit source/config changes before recording a run.\n" + status)
    config_data = args.config.read_bytes()
    config = json.loads(config_data)
    required = {"experiment_id", "model", "decision_protocol", "max_steps",
                "history_images", "datasets", "runtime", "prompt", "seed", "launch_command"}
    missing = required - config.keys()
    if missing:
        raise SystemExit("Missing effective config fields: " + ", ".join(sorted(missing)))
    if config["decision_protocol"] not in {
        "generated_action_reason_json", "action_token_logits_argmax", "mcts",
    }:
        raise SystemExit("Unknown decision protocol")
    for field in ("max_steps", "history_images"):
        if type(config[field]) is not int or config[field] <= 0:
            raise SystemExit(field + " must be a positive integer")
    if not isinstance(config["datasets"], list) or not config["datasets"]:
        raise SystemExit("datasets must be a nonempty list of repository-relative paths")
    datasets = []
    for name in config["datasets"]:
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise SystemExit("Dataset path must be inside the repository")
        data = path.read_bytes()
        datasets.append({"path": name, "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {
        "schema_version": 1, "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"),
        "git_dirty": False, "config": config,
        "config_sha256": hashlib.sha256(config_data).hexdigest(), "datasets": datasets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Recorded {config['experiment_id']} at commit {manifest['git_commit']}")


if __name__ == "__main__":
    main()
