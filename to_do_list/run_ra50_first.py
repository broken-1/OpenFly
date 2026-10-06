"""Run the standalone planner/tracker flow on RA50 sample 0.

This script does not start AirSim, execute actions, or import the MCTS evaluator.
The tracker image defaults to test/cur_img.jpg because the RA50 JSON contains poses
and instructions but not camera frames. The output records that image source.
"""

from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from instruction_progress import (
    InstructionProgressController,
    build_tracker_prompt,
    parse_tracker_update,
)
from qwen_json_generator import QwenGeneratorConfig, QwenJsonGenerator


def main():
    eval_info = Path(
        os.environ.get(
            "OPENFLY_EVAL_INFO",
            "configs/seen_airsim_23_random50_seed20260609.json",
        )
    )
    image_path = Path(
        os.environ.get("OPENFLY_TODO_IMAGE", "test/cur_img.jpg")
    )
    with eval_info.open(encoding="utf-8") as handle:
        item = json.load(handle)[0]

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
    state = controller.create_plan(item["gpt_instruction"])

    tracker_prompt = build_tracker_prompt(
        state,
        step=0,
        recent_actions=[],
    )
    raw_tracker = generator(tracker_prompt, [image_path])
    tracker_update = parse_tracker_update(raw_tracker)
    transition = state.apply(tracker_update, step=0)

    result = {
        "sample_idx": 0,
        "image_source": str(image_path),
        "image_is_sample_zero_frame": False,
        "instruction": item["gpt_instruction"],
        "plan": state.plan.to_dict(),
        "tracker_update": asdict(tracker_update),
        "transition": asdict(transition),
        "state": state.to_dict(),
    }
    output_path = Path(
        os.environ.get(
            "OPENFLY_TODO_RESULT",
            f"to_do_list/outputs/ra50_sample0_{time.strftime('%Y%m%d_%H%M%S')}.json",
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
