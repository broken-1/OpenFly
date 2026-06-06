import base64
import json
import math
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
try:
    from openai import OpenAI
except ImportError as exc:
    raise ImportError(
        "The OpenFly MCTS evaluator requires the openai package. "
        "Install it with: pip install 'openai>=1.58.1'"
    ) from exc

from eval import (
    AirsimBridge,
    GSBridge,
    UEBridge,
    AutoModelForVision2Seq,
    AutoProcessor,
    calculate_distance,
    convert_to_action_id,
    getPoseAfterMakeAction,
    get_images,
    kill_env_process,
    torch,
)


ACTION_NAMES = {
    0: "STOP",
    1: "FORWARD_3",
    2: "TURN_LEFT_30",
    3: "TURN_RIGHT_30",
    4: "UP_3",
    5: "DOWN_3",
    6: "LEFT_3",
    7: "RIGHT_3",
    8: "FORWARD_6",
    9: "FORWARD_9",
}
ACTION_IDS = {name: action_id for action_id, name in ACTION_NAMES.items()}
ACTION_VECTORS = {
    0: np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    1: np.array([0, 3, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    2: np.array([0, 0, 15, 0, 0, 0, 0, 0], dtype=np.float32),
    3: np.array([0, 0, 0, 15, 0, 0, 0, 0], dtype=np.float32),
    4: np.array([0, 0, 0, 0, 2, 0, 0, 0], dtype=np.float32),
    5: np.array([0, 0, 0, 0, 0, 2, 0, 0], dtype=np.float32),
    6: np.array([0, 0, 0, 0, 0, 0, 5, 0], dtype=np.float32),
    7: np.array([0, 0, 0, 0, 0, 0, 0, 5], dtype=np.float32),
    8: np.array([0, 6, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    9: np.array([0, 9, 0, 0, 0, 0, 0, 0], dtype=np.float32),
}
ACTION_ALIASES = {
    "STOP": 0,
    "FORWARD": 1,
    "MOVE_FORWARD": 1,
    "GO_FORWARD": 1,
    "FORWARD_3": 1,
    "MOVE_FORWARD_3": 1,
    "TURN_LEFT": 2,
    "LEFT_TURN": 2,
    "TURN_LEFT_30": 2,
    "TURN_RIGHT": 3,
    "RIGHT_TURN": 3,
    "TURN_RIGHT_30": 3,
    "UP": 4,
    "GO_UP": 4,
    "UP_3": 4,
    "DOWN": 5,
    "GO_DOWN": 5,
    "DOWN_3": 5,
    "LEFT": 6,
    "MOVE_LEFT": 6,
    "LEFT_3": 6,
    "RIGHT": 7,
    "MOVE_RIGHT": 7,
    "RIGHT_3": 7,
    "FORWARD_6": 8,
    "MOVE_FORWARD_6": 8,
    "FORWARD_9": 9,
    "MOVE_FORWARD_9": 9,
}
TOP_K_CANDIDATES = int(os.environ.get("OPENFLY_MCTS_TOP_K", "3"))
SEARCH_DEPTH = int(os.environ.get("OPENFLY_MCTS_SEARCH_DEPTH", "3"))
NUM_SIMULATIONS = int(os.environ.get("OPENFLY_MCTS_NUM_SIMULATIONS", "12"))
PUCT_C = float(os.environ.get("OPENFLY_MCTS_PUCT_C", "1.2"))
MAX_STEP = int(os.environ.get("OPENFLY_MCTS_MAX_STEP", "100"))
MAX_SAMPLES = int(os.environ.get("OPENFLY_MCTS_MAX_SAMPLES", "0"))
STOP_SELECT_VALUE = float(os.environ.get("OPENFLY_MCTS_STOP_SELECT_VALUE", "0.75"))
FALLBACK_TIE_EPS = float(os.environ.get("OPENFLY_MCTS_FALLBACK_TIE_EPS", "0.05"))
FALLBACK_OVERRIDE_MARGIN = float(
    os.environ.get("OPENFLY_MCTS_FALLBACK_OVERRIDE_MARGIN", "0.15")
)
STOP_FALLBACK_OVERRIDE_MARGIN = float(
    os.environ.get("OPENFLY_MCTS_STOP_FALLBACK_OVERRIDE_MARGIN", "0.30")
)
TRUST_FALLBACK_STOP = os.environ.get(
    "OPENFLY_MCTS_TRUST_FALLBACK_STOP", "1"
).lower() not in {"0", "false", "no"}
NEGATIVE_REASON_VALUE_CAP = float(
    os.environ.get("OPENFLY_MCTS_NEGATIVE_REASON_VALUE_CAP", "0.25")
)
OBS_ROOT = Path(os.environ.get("OPENFLY_MCTS_OBS_DIR", "test/mcts_obs"))
RESULTS_PATH = Path(
    os.environ.get("OPENFLY_MCTS_RESULTS", "outputs/eval_mcts_results.jsonl")
)

NEGATIVE_REASON_PATTERNS = (
    "contradict",
    "away from",
    "collision risk",
    "high risk",
    "misalign",
    "misaligned",
    "unstable",
    "unpromising",
    "failure",
    "wrong",
    "not appropriate",
    "premature",
    "obstruct",
    "crash",
    "black",
)


def clean_json_text(raw_text):
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def extract_json_payload(raw_text):
    cleaned = clean_json_text(raw_text)
    try:
        return json.loads(cleaned), cleaned
    except Exception:
        match = re.search(r"(\{.*\})", cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group(1)), cleaned
        raise


def normalize_action(action_value):
    if action_value is None:
        return None
    if isinstance(action_value, (int, np.integer)):
        return int(action_value) if int(action_value) in ACTION_NAMES else None

    text = unicodedata.normalize("NFKC", str(action_value)).strip().upper()
    text = re.sub(r"[\s\-]+", "_", text)
    if text.isdigit():
        action_id = int(text)
        return action_id if action_id in ACTION_NAMES else None
    return ACTION_ALIASES.get(text)


def clamp_score(value, default=50.0):
    try:
        score = float(value)
    except Exception:
        score = default
    return max(0.0, min(100.0, score))


def has_negative_reason(reason):
    text = unicodedata.normalize("NFKC", str(reason)).lower()
    return any(pattern in text for pattern in NEGATIVE_REASON_PATTERNS)


def action_names(action_ids):
    return [ACTION_NAMES[int(action_id)] for action_id in action_ids]


def count_trailing_actions(actions, action_set):
    count = 0
    for action_id in reversed(actions):
        if int(action_id) not in action_set:
            break
        count += 1
    return count


def signed_angle_delta_deg(current_yaw, start_yaw):
    delta = math.degrees(float(current_yaw) - float(start_yaw))
    return ((delta + 180.0) % 360.0) - 180.0


def build_history_context(acts, start_pose, root_pose):
    recent_actions = acts[-10:]
    return {
        "step_index": len(acts),
        "executed_actions": action_names(recent_actions),
        "last_5_actions": action_names(acts[-5:]),
        "consecutive_forward_count": count_trailing_actions(acts, {1, 8, 9}),
        "consecutive_turn_count": count_trailing_actions(acts, {2, 3}),
        "distance_moved_from_start_m": round(
            calculate_distance(start_pose[:3], root_pose[:3]), 3
        ),
        "yaw_change_from_start_deg": round(
            signed_angle_delta_deg(root_pose[3], start_pose[3]), 3
        ),
    }


def compute_leaf_value(prm_eval):
    score = clamp_score(prm_eval.get("score")) / 100.0
    progress = clamp_score(prm_eval.get("progress_score")) / 100.0
    alignment = clamp_score(prm_eval.get("alignment_score")) / 100.0
    safety = 1.0 - clamp_score(prm_eval.get("risk_score")) / 100.0

    # The judge can occasionally assign a high holistic score while its own
    # explanation says the action is risky or contradicts the instruction.
    value = 0.45 * score + 0.20 * progress + 0.25 * alignment + 0.10 * safety
    if has_negative_reason(prm_eval.get("reason", "")):
        value = min(value, NEGATIVE_REASON_VALUE_CAP)
    return max(0.0, min(1.0, value))


def prepare_policy_inputs(processor, image_list, text):
    policy_images = get_images(image_list, True, 2)
    if isinstance(policy_images, np.ndarray):
        img = Image.fromarray(policy_images)
        images = [img, img, img]
    else:
        images = [Image.fromarray(img) for img in policy_images]
    return processor(text, images).to("cuda:0", dtype=torch.bfloat16)


def action_vector_to_token_ids(policy, action_vector, unnorm_key="vlnv1"):
    action_stats = policy.get_action_stats(unnorm_key)
    mask = action_stats.get(
        "mask", np.ones_like(action_stats["q01"], dtype=bool)
    )
    action_high = np.array(action_stats["q99"], dtype=np.float32)
    action_low = np.array(action_stats["q01"], dtype=np.float32)

    normalized = np.array(action_vector, dtype=np.float32)
    denom = np.maximum(action_high - action_low, 1e-6)
    normalized = np.where(mask, 2 * (normalized - action_low) / denom - 1, normalized)
    normalized = np.clip(normalized, -1.0, 1.0)

    bin_indices = np.abs(policy.bin_centers[:, None] - normalized[None, :]).argmin(axis=0)
    token_ids = policy.vocab_size - (bin_indices + 1)
    return token_ids.astype(np.int64)


def score_action_tokens(policy, inputs, action_token_ids):
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if not torch.all(input_ids[:, -1] == 29871):
        spacer = torch.tensor([[29871]], device=input_ids.device, dtype=input_ids.dtype)
        input_ids = torch.cat((input_ids, spacer), dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype),
                ),
                dim=1,
            )

    candidate_ids = torch.tensor(
        action_token_ids[None, :], device=input_ids.device, dtype=input_ids.dtype
    )
    full_input_ids = torch.cat((input_ids, candidate_ids), dim=1)
    if attention_mask is None:
        full_attention_mask = torch.ones_like(full_input_ids)
    else:
        full_attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones(candidate_ids.shape, device=attention_mask.device, dtype=attention_mask.dtype),
            ),
            dim=1,
        )

    with torch.inference_mode():
        outputs = policy(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            pixel_values=inputs["pixel_values"],
            return_dict=True,
        )

    logits = outputs.logits[0, -candidate_ids.shape[1] - 1 : -1, :]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_scores = log_probs[
        torch.arange(candidate_ids.shape[1], device=log_probs.device),
        candidate_ids[0].long(),
    ]
    return float(token_scores.sum().item())


def get_openfly_action_topk(policy, processor, image_list, text, top_k):
    inputs = prepare_policy_inputs(processor, image_list, text)
    with torch.inference_mode():
        generated_action = policy.predict_action(
            **inputs, unnorm_key="vlnv1", do_sample=False
        )
    generated_action_id = convert_to_action_id(
        generated_action.round().astype(int)
    )

    scored = []
    for action_id, action_vector in ACTION_VECTORS.items():
        token_ids = action_vector_to_token_ids(policy, action_vector)
        log_prob = score_action_tokens(policy, inputs, token_ids)
        scored.append((action_id, log_prob))

    scored.sort(key=lambda item: item[1], reverse=True)
    selected_ids = [generated_action_id]
    for action_id, _ in scored:
        if action_id not in selected_ids:
            selected_ids.append(action_id)
        if len(selected_ids) >= max(1, top_k):
            break

    score_by_action = dict(scored)
    top = [(action_id, score_by_action[action_id]) for action_id in selected_ids]
    best_log_prob = max(score for _, score in top)
    min_log_prob = min(score for _, score in top)
    span = max(best_log_prob - min_log_prob, 1e-6)

    candidates = []
    for rank, (action_id, log_prob) in enumerate(top):
        if len(top) == 1:
            score = 100.0
        else:
            score = 50.0 + 50.0 * (log_prob - min_log_prob) / span
        if action_id == generated_action_id:
            score = max(score, 100.0)
        candidates.append(
            {
                "action": action_id,
                "action_name": ACTION_NAMES[action_id],
                "score": clamp_score(score),
                "log_prob": log_prob,
                "reason": (
                    "OpenFly generated baseline action."
                    if action_id == generated_action_id and rank == 0
                    else "OpenFly local policy top-k candidate."
                ),
            }
        )

    raw = {
        "source": "openfly_policy_topk",
        "top_k": top_k,
        "generated_action": generated_action.tolist(),
        "generated_action_id": generated_action_id,
        "generated_action_name": ACTION_NAMES[generated_action_id],
        "ranked_actions": [
            {
                "action": action_id,
                "action_name": ACTION_NAMES[action_id],
                "log_prob": log_prob,
            }
            for action_id, log_prob in scored
        ],
    }
    return candidates, top[0][0], json.dumps(raw, ensure_ascii=False)


def build_fallback_neighborhood_candidates(fallback_action, limit=TOP_K_CANDIDATES):
    fallback_action = int(fallback_action)
    neighbor_map = {
        0: [0, 1, 8, 2, 3],
        1: [1, 8, 9, 2, 3, 0],
        2: [2, 1, 8, 9, 0],
        3: [3, 1, 8, 9, 0],
        4: [4, 1, 2, 3, 0],
        5: [5, 1, 2, 3, 0],
        6: [6, 1, 2, 3, 0],
        7: [7, 1, 2, 3, 0],
        8: [8, 1, 9, 2, 3, 0],
        9: [9, 8, 1, 2, 3, 0],
    }
    action_ids = neighbor_map.get(fallback_action, [fallback_action, 0])
    if limit is not None:
        action_ids = action_ids[:limit]
    candidates = []
    for rank, action_id in enumerate(action_ids):
        candidates.append(
            {
                "action": action_id,
                "action_name": ACTION_NAMES[action_id],
                "score": max(50.0, 100.0 - rank * 10.0),
                "reason": "Local fallback neighborhood candidate.",
            }
        )
    return candidates


def merge_candidate_lists(*candidate_lists):
    merged = []
    seen = set()
    for candidates in candidate_lists:
        for item in candidates:
            action_id = int(item["action"])
            if action_id in seen:
                continue
            seen.add(action_id)
            merged.append(item)
    return merged


def capture_current_image(
    env_bridge,
    pose,
    pos_ratio,
    pitch,
    output_dir,
    restore_pose=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    env_bridge.set_camera_pose(
        pose[0] / pos_ratio,
        pose[1] / pos_ratio,
        pose[2] / pos_ratio,
        pitch,
        np.rad2deg(pose[3]),
        0,
    )
    time.sleep(0.05)
    image = env_bridge.get_camera_data()
    image_path = output_dir / "current.png"
    cv2.imwrite(str(image_path), image)

    if restore_pose is not None:
        env_bridge.set_camera_pose(
            restore_pose[0] / pos_ratio,
            restore_pose[1] / pos_ratio,
            restore_pose[2] / pos_ratio,
            pitch,
            np.rad2deg(restore_pose[3]),
            0,
        )

    return str(image_path), image


class VisionJudge:
    def __init__(self):
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY is required for OpenFly MCTS external VL scoring."
            )
        self.model_name = os.environ.get("OPENFLY_MCTS_JUDGE_MODEL", "qwen3.7-plus")
        base_url = os.environ.get(
            "OPENFLY_MCTS_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        client_kwargs = {"api" + "_key": api_key, "base_url": base_url}
        self.client = OpenAI(**client_kwargs)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def query_image(self, prompt, image_path):
        return self.query_images(prompt, [("Current camera image", image_path)])

    def query_images(self, prompt, image_items):
        content = []
        for label, image_path in image_items:
            content.append({"type": "text", "text": f"{label}:"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self.encode_image(image_path)}"
                    },
                }
            )
        content.append({"type": "text", "text": prompt})

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            extra_body={"enable_thinking": False},
        )
        return unicodedata.normalize("NFKC", completion.choices[0].message.content)


def build_candidate_prompt(instruction_text):
    action_space = ", ".join(f"{idx}:{name}" for idx, name in ACTION_NAMES.items())
    return f"""
You are controlling a drone in an aerial visual navigation task.
Use the current camera image to propose the top {TOP_K_CANDIDATES} next actions.

Navigation instruction:
{instruction_text}

Action space:
{action_space}

Important:
- If the drone appears to have reached the destination described by the instruction,
  include STOP as a strong candidate.
- If the OpenFly policy action is provided as a fallback by the system, it will be
  considered together with your candidates, so focus on visual navigation quality.

Return JSON only:
{{
  "candidates": [
    {{"action": "ACTION_NAME_OR_ID", "score": 0-100, "reason": "short reason"}},
    {{"action": "ACTION_NAME_OR_ID", "score": 0-100, "reason": "short reason"}},
    {{"action": "ACTION_NAME_OR_ID", "score": 0-100, "reason": "short reason"}}
  ]
}}
""".strip()


def build_prm_prompt(instruction_text, action_prefix, history_context, available_actions):
    action_sequence = " -> ".join(
        ACTION_NAMES[action_id] for action_id in action_prefix
    ) if action_prefix else "NONE"
    first_action = ACTION_NAMES[action_prefix[0]] if action_prefix else "NONE"
    history_json = json.dumps(history_context or {}, ensure_ascii=False, indent=2)
    available_action_text = ", ".join(
        ACTION_NAMES[int(action_id)] for action_id in available_actions
    ) if available_actions else "UNKNOWN"
    return f"""
You are evaluating a short-horizon drone navigation rollout.

Navigation instruction:
{instruction_text}

Visual memory is provided in strict chronological order:
Image 1: start_view, the first observation before any action.
Image 2: current_view, the real current state before evaluating the candidate.
Image 3: candidate_leaf_view, the state after executing the candidate sequence.

Navigation history summary:
{history_json}

Available root actions being compared now:
{available_action_text}

Candidate action sequence:
{action_sequence}

Root action being compared now:
{first_action}

The image shows the leaf state after executing the candidate sequence.
Judge whether the ROOT ACTION is the right action to take NOW, compared with
other available actions such as STOP, FORWARD, TURN_LEFT, TURN_RIGHT, UP, DOWN,
LEFT, and RIGHT.

Important scoring rules:
- Do NOT give a high score merely because the state may eventually be useful.
- If the state is close enough to the described target and stopping would be
  appropriate, STOP should score high and moving should score low.
- If the next required maneuver is turning or re-aligning, a forward-only action
  should not receive a high alignment score just because a later turn may help.
- Penalize overshooting, drifting past the target, repeated forward movement
  without resolving the instruction, and actions that postpone the needed turn.
- Higher risk means the rollout is unstable, oscillatory, likely overshooting, or
  likely moving away from the target.
- Use the image order carefully. Compare Image 2 to Image 3 to judge whether the
  candidate root action improves the current state or merely repeats a stale
  movement pattern.

Return JSON only:
{{
  "score": 0-100,
  "progress_score": 0-100,
  "alignment_score": 0-100,
  "risk_score": 0-100,
  "reason": "short reason"
}}
""".strip()


def parse_candidate_response(raw_text):
    payload, cleaned = extract_json_payload(raw_text)
    raw_candidates = payload.get("candidates", [])
    candidates = []
    seen = set()
    for item in raw_candidates:
        action_id = normalize_action(item.get("action"))
        if action_id is None or action_id in seen:
            continue
        seen.add(action_id)
        candidates.append(
            {
                "action": action_id,
                "action_name": ACTION_NAMES[action_id],
                "score": clamp_score(item.get("score", 50.0)),
                "reason": str(item.get("reason", "")).strip(),
            }
        )
    return candidates[:TOP_K_CANDIDATES], cleaned


def parse_prm_response(raw_text):
    default_eval = {
        "score": 50.0,
        "progress_score": 50.0,
        "alignment_score": 50.0,
        "risk_score": 50.0,
        "reason": f"[PRM_PARSE_FAILED] {clean_json_text(raw_text)}",
    }
    try:
        payload, cleaned = extract_json_payload(raw_text)
    except Exception:
        return default_eval, clean_json_text(raw_text)

    result = {
        "score": clamp_score(payload.get("score"), default_eval["score"]),
        "progress_score": clamp_score(
            payload.get("progress_score"), default_eval["progress_score"]
        ),
        "alignment_score": clamp_score(
            payload.get("alignment_score"), default_eval["alignment_score"]
        ),
        "risk_score": clamp_score(payload.get("risk_score"), default_eval["risk_score"]),
        "reason": str(payload.get("reason", "")).strip() or default_eval["reason"],
    }
    return result, cleaned


def query_action_candidates(judge, instruction_text, image_path, fallback_action):
    prompt = build_candidate_prompt(instruction_text)
    try:
        raw_response = judge.query_image(prompt, image_path)
        candidates, cleaned_response = parse_candidate_response(raw_response)
        if candidates:
            return merge_fallback_candidate(candidates, fallback_action), cleaned_response
    except Exception as exc:
        cleaned_response = f"[CANDIDATE_QUERY_FAILED] {exc}"

    return merge_fallback_candidate([
        {
            "action": int(fallback_action),
            "action_name": ACTION_NAMES[int(fallback_action)],
            "score": 50.0,
            "reason": "Fallback to OpenFly single-step action.",
        }
    ], fallback_action), cleaned_response


def merge_fallback_candidate(candidates, fallback_action):
    fallback_action = int(fallback_action)
    merged = []
    seen = set()
    for item in candidates:
        action_id = int(item["action"])
        if action_id in seen:
            continue
        seen.add(action_id)
        merged.append(item)

    if fallback_action not in seen:
        merged.append(
            {
                "action": fallback_action,
                "action_name": ACTION_NAMES[fallback_action],
                "score": 80.0,
                "reason": "OpenFly policy fallback action, kept as a baseline-safe candidate.",
            }
        )

    if len(merged) == 1 and merged[0]["action"] == 0:
        for action_id, score in [(1, 55.0), (8, 50.0)]:
            merged.append(
                {
                    "action": action_id,
                    "action_name": ACTION_NAMES[action_id],
                    "score": score,
                    "reason": "Safety candidate added because STOP was the only proposed action.",
                }
            )

    return merged


def query_prm_score(
    judge,
    instruction_text,
    image_items,
    action_prefix,
    history_context,
    available_actions,
):
    prompt = build_prm_prompt(
        instruction_text,
        action_prefix,
        history_context,
        available_actions,
    )
    try:
        raw_response = judge.query_images(prompt, image_items)
        return parse_prm_response(raw_response)
    except Exception as exc:
        fallback = {
            "score": 50.0,
            "progress_score": 50.0,
            "alignment_score": 50.0,
            "risk_score": 50.0,
            "reason": f"[PRM_QUERY_FAILED] {exc}",
        }
        return fallback, str(exc)


@dataclass
class SearchNode:
    pose: list
    step_idx: int
    action_prefix: list
    prior_score: float
    parent: "SearchNode" = None
    children: dict = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    leaf_eval: dict = None
    candidate_actions: list = field(default_factory=list)
    candidate_priors: dict = field(default_factory=dict)
    candidate_raw: str = ""

    @property
    def mean_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def select_child_by_puct(node):
    best_child = None
    best_score = None
    parent_visits = max(1, node.visit_count)

    for child in node.children.values():
        q_score = child.mean_value
        u_score = PUCT_C * child.prior_score * math.sqrt(parent_visits) / (
            1 + child.visit_count
        )
        total_score = q_score + u_score
        if best_score is None or total_score > best_score:
            best_score = total_score
            best_child = child
    return best_child


def expand_child(node, action_id, prior_score):
    child = SearchNode(
        pose=getPoseAfterMakeAction(node.pose, action_id),
        step_idx=node.step_idx + 1,
        action_prefix=node.action_prefix + [action_id],
        prior_score=prior_score,
        parent=node,
    )
    node.children[action_id] = child
    return child


def build_root_action_stats(root):
    root_action_stats = []
    for action_id, child in root.children.items():
        root_action_stats.append(
            {
                "action": action_id,
                "action_name": ACTION_NAMES[action_id],
                "visit_count": child.visit_count,
                "mean_value": child.mean_value,
                "prior_score": child.prior_score,
            }
        )
    root_action_stats.sort(
        key=lambda item: (item["mean_value"], item["visit_count"], item["prior_score"]),
        reverse=True,
    )
    return root_action_stats


def choose_best_root_action(root_action_stats, fallback_action):
    if not root_action_stats:
        return None

    fallback_action = int(fallback_action)
    eligible = list(root_action_stats)

    if fallback_action == 0:
        if TRUST_FALLBACK_STOP:
            return 0
        stop_item = next((item for item in eligible if item["action"] == 0), None)
        if stop_item is not None:
            best_mean = max(item["mean_value"] for item in eligible)
            if best_mean - stop_item["mean_value"] <= STOP_FALLBACK_OVERRIDE_MARGIN:
                return 0

    # STOP is easy to over-predict from a single image. Only allow it when the
    # rollout scorer is genuinely confident, unless it is the only action left.
    if len(eligible) > 1 and fallback_action != 0:
        non_stop = [
            item
            for item in eligible
            if item["action"] != 0 or item["mean_value"] >= STOP_SELECT_VALUE
        ]
        if non_stop:
            eligible = non_stop

    best_mean = max(item["mean_value"] for item in eligible)
    fallback_item = next(
        (item for item in eligible if item["action"] == fallback_action), None
    )
    if fallback_item is not None:
        # Keep MCTS baseline-safe: only override OpenFly's own action when the
        # rollout judge finds a clearly better root action.
        if best_mean - fallback_item["mean_value"] <= FALLBACK_OVERRIDE_MARGIN:
            return fallback_action

    near_best = [
        item for item in eligible if best_mean - item["mean_value"] <= FALLBACK_TIE_EPS
    ]

    for item in near_best:
        if item["action"] == fallback_action:
            return item["action"]

    near_best.sort(
        key=lambda item: (item["mean_value"], item["prior_score"], item["visit_count"]),
        reverse=True,
    )
    return near_best[0]["action"]


def ensure_node_candidates(
    judge,
    env_bridge,
    instruction_text,
    node,
    root_pose,
    pos_ratio,
    pitch,
    obs_dir,
    fallback_action,
):
    if node.candidate_actions:
        return
    if node.action_prefix and node.action_prefix[-1] == 0:
        node.candidate_actions = [0]
        node.candidate_priors = {0: 1.0}
        return

    image_path, _ = capture_current_image(
        env_bridge,
        node.pose,
        pos_ratio,
        pitch,
        obs_dir,
        restore_pose=root_pose,
    )
    candidates = build_fallback_neighborhood_candidates(fallback_action)
    raw_response = json.dumps(
        {
            "source": "local_fallback_neighborhood",
            "fallback_action": int(fallback_action),
            "fallback_action_name": ACTION_NAMES[int(fallback_action)],
        },
        ensure_ascii=False,
    )
    node.candidate_actions = [item["action"] for item in candidates]
    node.candidate_priors = {
        item["action"]: max(0.01, item["score"] / 100.0) for item in candidates
    }
    node.candidate_raw = raw_response


def evaluate_leaf_node(
    judge,
    env_bridge,
    instruction_text,
    node,
    root_pose,
    pos_ratio,
    pitch,
    obs_dir,
    start_image_path,
    current_image_path,
    history_context,
    available_actions,
):
    leaf_image_path, _ = capture_current_image(
        env_bridge,
        node.pose,
        pos_ratio,
        pitch,
        obs_dir,
        restore_pose=root_pose,
    )
    image_items = [
        ("Image 1 - start_view", start_image_path),
        ("Image 2 - current_view", current_image_path),
        ("Image 3 - candidate_leaf_view", leaf_image_path),
    ]
    prm_eval, prm_raw = query_prm_score(
        judge,
        instruction_text,
        image_items,
        node.action_prefix,
        history_context,
        available_actions,
    )
    node.leaf_eval = {
        "prm": prm_eval,
        "raw_response": prm_raw,
        "image_order": [label for label, _ in image_items],
        "image_paths": [path for _, path in image_items],
    }
    return compute_leaf_value(prm_eval)


def run_mcts_search(
    judge,
    env_bridge,
    instruction_text,
    root_pose,
    root_candidates,
    pos_ratio,
    pitch,
    obs_dir,
    fallback_action,
    start_image_path,
    current_image_path,
    history_context,
):
    root = SearchNode(
        pose=list(root_pose),
        step_idx=0,
        action_prefix=[],
        prior_score=1.0,
        candidate_actions=[item["action"] for item in root_candidates],
        candidate_priors={
            item["action"]: max(0.01, item["score"] / 100.0)
            for item in root_candidates
        },
    )

    simulation_logs = []
    simulation_count = max(NUM_SIMULATIONS, len(root_candidates))
    for sim_idx in range(simulation_count):
        node = root
        path = [root]

        while len(node.action_prefix) < SEARCH_DEPTH:
            if node.action_prefix and node.action_prefix[-1] == 0:
                break
            ensure_node_candidates(
                judge,
                env_bridge,
                instruction_text,
                node,
                root_pose,
                pos_ratio,
                pitch,
                obs_dir / f"sim_{sim_idx + 1}_node_{len(node.action_prefix)}",
                fallback_action,
            )
            unexpanded = [
                action for action in node.candidate_actions if action not in node.children
            ]
            if unexpanded:
                action_id = unexpanded[0]
                node = expand_child(node, action_id, node.candidate_priors[action_id])
                path.append(node)
                break

            selected = select_child_by_puct(node)
            if selected is None:
                break
            node = selected
            path.append(node)

        leaf_value = evaluate_leaf_node(
            judge,
            env_bridge,
            instruction_text,
            node,
            root_pose,
            pos_ratio,
            pitch,
            obs_dir / f"sim_{sim_idx + 1}_leaf",
            start_image_path,
            current_image_path,
            history_context,
            root.candidate_actions,
        )

        for path_node in path:
            path_node.visit_count += 1
            path_node.value_sum += leaf_value

        prm = node.leaf_eval["prm"]
        simulation_logs.append(
            {
                "simulation": sim_idx + 1,
                "action_prefix": node.action_prefix,
                "action_prefix_names": [ACTION_NAMES[a] for a in node.action_prefix],
                "score": prm["score"],
                "progress_score": prm["progress_score"],
                "alignment_score": prm["alignment_score"],
                "risk_score": prm["risk_score"],
                "leaf_value": leaf_value,
                "reason": prm["reason"],
                "image_order": node.leaf_eval.get("image_order", []),
                "image_paths": node.leaf_eval.get("image_paths", []),
            }
        )

    if not root.children:
        return root_candidates[0]["action"], simulation_logs, root, []

    root_action_stats = build_root_action_stats(root)
    best_action = choose_best_root_action(root_action_stats, fallback_action)
    return best_action, simulation_logs, root, root_action_stats


def load_eval_info(eval_path):
    with open(eval_path, "r") as f:
        all_eval_info = json.load(f)
    if MAX_SAMPLES > 0:
        return all_eval_info[:MAX_SAMPLES]
    return all_eval_info


def build_env_groups(all_eval_info):
    env_groups = {}
    for item in all_eval_info:
        env_name = item["image_path"].split("/")[0]
        env_groups.setdefault(env_name, []).append(item)
    return env_groups


def create_env_bridge(env_name):
    if "airsim" in env_name:
        return AirsimBridge(env_name), 1.0
    if "ue" in env_name:
        return UEBridge(ue_ip="127.0.0.1", ue_port="9000", env_name=env_name), 1.0
    if "gs" in env_name:
        return GSBridge(env_name), 5.15
    raise ValueError(f"Unknown environment type: {env_name}")


def append_jsonl(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def cleanup_env():
    for keyword in ["AirVLN", "guangzhou", "shanghai", "CitySample", "CrashReport"]:
        kill_env_process(keyword)


def main():
    eval_info = os.environ.get("OPENFLY_EVAL_INFO", "configs/eval_bigcity.json")
    model_name_or_path = os.environ.get(
        "OPENFLY_MODEL_PATH", "/home/liusongbo/models/openfly-agent-7b"
    )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("", encoding="utf-8")

    judge = VisionJudge()
    all_eval_info = load_eval_info(eval_info)

    processor = AutoProcessor.from_pretrained(model_name_or_path)
    policy = AutoModelForVision2Seq.from_pretrained(
        model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda:0")

    acc = 0
    stop_count = 0
    data_num = 0
    scene_metrics = {}
    env_groups = build_env_groups(all_eval_info)

    print(
        "MCTS settings: "
        f"top_k={TOP_K_CANDIDATES}, depth={SEARCH_DEPTH}, "
        f"simulations={NUM_SIMULATIONS}, puct_c={PUCT_C}"
    )
    print(f"MCTS results log: {RESULTS_PATH}")

    for env_name, group_eval_info in env_groups.items():
        print(
            f"Starting MCTS evaluation of environment: {env_name}, "
            f"with {len(group_eval_info)} data entries"
        )
        time.sleep(5)
        try:
            env_bridge, pos_ratio = create_env_bridge(env_name)
        except ValueError as exc:
            print(exc)
            continue

        for idx, item in enumerate(group_eval_info):
            acts = []
            data_num += 1
            pos_list = item["pos"]
            text = item["gpt_instruction"]
            start_position = pos_list[0]
            start_yaw = item["yaw"][0]
            new_pose = [
                start_position[0],
                start_position[1],
                start_position[2],
                start_yaw,
            ]
            end_position = pos_list[-1]
            pitch = -45.0 if "high" in item["image_path"] else 0.0

            print(
                f"Sample {idx}: {start_position} -> {end_position}, "
                f"initial heading: {start_yaw}"
            )
            env_bridge.set_camera_pose(
                new_pose[0] / pos_ratio,
                new_pose[1] / pos_ratio,
                new_pose[2] / pos_ratio,
                pitch,
                np.rad2deg(new_pose[3]),
                0,
            )

            step = 0
            flag_osr = 0
            image_list = []
            image_error = False
            env_bridge.pass_len = 1e-3
            old_pose = list(new_pose)
            start_pose = list(new_pose)
            sample_obs_dir = OBS_ROOT / env_name / f"sample_{idx}"
            start_image_path, _ = capture_current_image(
                env_bridge,
                start_pose,
                pos_ratio,
                pitch,
                sample_obs_dir / "start",
                restore_pose=start_pose,
            )

            while step < MAX_STEP:
                try:
                    root_pose = list(new_pose)
                    obs_dir = sample_obs_dir / f"step_{step}"
                    current_image_path, current_image = capture_current_image(
                        env_bridge,
                        root_pose,
                        pos_ratio,
                        pitch,
                        obs_dir / "root",
                        restore_pose=root_pose,
                    )
                    image_list.append(current_image)
                    history_context = build_history_context(
                        acts,
                        start_pose,
                        root_pose,
                    )

                    root_candidates, fallback_action, candidate_raw = get_openfly_action_topk(
                        policy,
                        processor,
                        image_list,
                        text,
                        TOP_K_CANDIDATES,
                    )
                    root_candidates = merge_candidate_lists(
                        root_candidates,
                        build_fallback_neighborhood_candidates(
                            fallback_action,
                            limit=None,
                        ),
                    )
                    candidate_desc = ", ".join(
                        f"{item['action_name']}({item['score']:.1f})"
                        for item in root_candidates
                    )
                    print(
                        f"[Sample {idx} Step {step}] candidates: {candidate_desc}"
                    )

                    (
                        selected_action,
                        simulation_logs,
                        root_node,
                        root_action_stats,
                    ) = run_mcts_search(
                        judge,
                        env_bridge,
                        text,
                        root_pose,
                        root_candidates,
                        pos_ratio,
                        pitch,
                        obs_dir,
                        fallback_action,
                        start_image_path,
                        current_image_path,
                        history_context,
                    )
                    acts.append(selected_action)
                    selected_child = root_node.children.get(selected_action)
                    selected_reason = ""
                    if selected_child and selected_child.leaf_eval:
                        selected_reason = selected_child.leaf_eval["prm"]["reason"]
                    else:
                        selected_reason = root_candidates[0].get("reason", "")

                    print(
                        f"[Sample {idx} Step {step}] selected: "
                        f"{ACTION_NAMES[selected_action]} ({selected_action})"
                    )
                    print(f"[Sample {idx} Step {step}] reason: {selected_reason}")

                    append_jsonl(
                        RESULTS_PATH,
                        {
                            "env_name": env_name,
                            "sample_idx": idx,
                            "global_sample_idx": data_num,
                            "step": step,
                            "instruction": text,
                            "root_pose": root_pose,
                            "fallback_action": fallback_action,
                            "fallback_action_name": ACTION_NAMES[fallback_action],
                            "candidates": root_candidates,
                            "candidate_raw": candidate_raw,
                            "history_context": history_context,
                            "visual_memory_order": [
                                "Image 1 - start_view",
                                "Image 2 - current_view",
                                "Image 3 - candidate_leaf_view",
                            ],
                            "start_image_path": start_image_path,
                            "current_image_path": current_image_path,
                            "simulations": simulation_logs,
                            "root_action_stats": root_action_stats,
                            "selected_action": selected_action,
                            "selected_action_name": ACTION_NAMES[selected_action],
                            "selected_reason": selected_reason,
                        },
                    )

                    env_bridge.set_camera_pose(
                        root_pose[0] / pos_ratio,
                        root_pose[1] / pos_ratio,
                        root_pose[2] / pos_ratio,
                        pitch,
                        np.rad2deg(root_pose[3]),
                        0,
                    )

                    if selected_action == 0:
                        stop_count += 1
                        break

                    new_pose = getPoseAfterMakeAction(new_pose, selected_action)
                    env_bridge.set_camera_pose(
                        new_pose[0] / pos_ratio,
                        new_pose[1] / pos_ratio,
                        new_pose[2] / pos_ratio,
                        pitch,
                        np.rad2deg(new_pose[3]),
                        0,
                    )
                    env_bridge.pass_len += calculate_distance(old_pose, new_pose)
                    dis = calculate_distance(end_position, new_pose)
                    if dis < 20 and flag_osr != 2:
                        flag_osr = 2
                        env_bridge.osr.append(1)
                    old_pose = list(new_pose)
                    step += 1
                except Exception as exc:
                    print(f"Error during MCTS step: {exc}")
                    image_error = True
                    break

            dis = calculate_distance(end_position, new_pose)
            env_bridge.traj_len = calculate_distance(end_position, start_position)
            env_bridge.distance_to_goal.append(dis)
            if dis < 20:
                env_bridge.success.append(1)
                shortest_path = max(env_bridge.traj_len, 1e-12)
                actual_path = max(env_bridge.pass_len, shortest_path)
                env_bridge.spl.append(shortest_path / actual_path)
                acc += 1
            else:
                env_bridge.success.append(0)
                env_bridge.spl.append(0)
            if flag_osr == 0:
                env_bridge.osr.append(1 if dis < 20 else 0)
            env_bridge.print_info()

            metrics = scene_metrics.setdefault(
                env_name,
                {
                    "count": 0,
                    "ne_sum": 0.0,
                    "sr_sum": 0.0,
                    "osr_sum": 0.0,
                    "spl_sum": 0.0,
                },
            )
            metrics["count"] += 1
            metrics["ne_sum"] += env_bridge.distance_to_goal[-1]
            metrics["sr_sum"] += env_bridge.success[-1]
            metrics["osr_sum"] += env_bridge.osr[-1]
            metrics["spl_sum"] += env_bridge.spl[-1]

            if image_error:
                continue

        print(f"Completed MCTS evaluation of environment {env_name}")
        cleanup_env()
        del env_bridge
        import gc

        gc.collect()

    final_acc = acc / data_num if data_num > 0 else 0
    final_stop = stop_count / data_num if data_num > 0 else 0

    print("\nEvaluation complete!")
    if scene_metrics:
        print(f"\n{'Scene':<20} {'NE/m':>10} {'SR/%':>10} {'OSR/%':>10} {'SPL/%':>10}")
        for env_name, metrics in scene_metrics.items():
            count = metrics["count"]
            mean_ne = metrics["ne_sum"] / count if count > 0 else 0
            mean_sr = metrics["sr_sum"] / count * 100 if count > 0 else 0
            mean_osr = metrics["osr_sum"] / count * 100 if count > 0 else 0
            mean_spl = metrics["spl_sum"] / count * 100 if count > 0 else 0
            print(
                f"{env_name:<20} {mean_ne:>10.2f} {mean_sr:>10.2f} "
                f"{mean_osr:>10.2f} {mean_spl:>10.2f}"
            )
    print(f"Total samples: {data_num}")
    print(f"Final accuracy: {final_acc:.4f}")
    print(f"Final stop rate: {final_stop:.4f}")
    print(f"MCTS results log: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
