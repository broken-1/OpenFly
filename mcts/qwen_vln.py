"""Shared Qwen3-VL policy and rollout-value backend for drone navigation."""

from dataclasses import asdict, dataclass
import json
from typing import Sequence

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


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
TURN_ACTIONS = (2, 3)

POLICY_PROMPT_VERSION = "qwen_action_logits_v3"
VALUE_PROMPT_VERSION = "qwen_task_success_value_logits_v3"


@dataclass(frozen=True)
class QwenConfig:
    model_path: str
    device: str = "cuda:0"
    attention: str = "flash_attention_2"
    history_images: int = 3


@dataclass(frozen=True)
class ActionCandidate:
    action: int
    probability: float
    prior: float


@dataclass(frozen=True)
class PolicyOutput:
    probabilities: tuple[float, ...]

    @property
    def action(self):
        return int(np.argmax(self.probabilities))

    def candidates(self, top_k: int) -> list[ActionCandidate]:
        """Return movement Top-K plus the best turn action."""
        ranked = sorted(
            (action for action in ACTION_NAMES if action != 0),
            key=self.probabilities.__getitem__,
            reverse=True,
        )
        selected = ranked[:top_k]
        best_turn = max(TURN_ACTIONS, key=self.probabilities.__getitem__)
        if best_turn not in selected:
            selected.append(best_turn)

        probability_sum = sum(self.probabilities[action] for action in selected)
        return [
            ActionCandidate(
                action=action,
                probability=self.probabilities[action],
                prior=self.probabilities[action] / probability_sum,
            )
            for action in selected
        ]

    def to_dict(self):
        return {
            "action": self.action,
            "action_name": ACTION_NAMES[self.action],
            "probabilities": {
                ACTION_NAMES[action]: probability
                for action, probability in enumerate(self.probabilities)
            },
            "prompt_version": POLICY_PROMPT_VERSION,
        }


@dataclass(frozen=True)
class TurnKeyframe:
    step: int
    action: int
    image: object

    def metadata(self):
        return {
            "step": self.step,
            "action": self.action,
            "action_name": ACTION_NAMES[self.action],
        }



@dataclass(frozen=True)
class RolloutEvaluation:
    success_probability: float
    failure_probability: float
    question_type: str
    real_image_count: int
    virtual_image_count: int

    @property
    def value(self):
        return self.success_probability

    def to_dict(self):
        payload = asdict(self)
        payload["prompt_version"] = VALUE_PROMPT_VERSION
        return payload


def _as_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")


class QwenVLNBackend:
    """One Qwen instance shared by the action policy and rollout evaluator."""

    def __init__(self, config: QwenConfig):
        self.config = config
        self.processor = AutoProcessor.from_pretrained(
            config.model_path,
            local_files_only=True,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.model_path,
            dtype=torch.bfloat16,
            attn_implementation=config.attention,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(config.device)
        self.model.eval()

        tokenizer = self.processor.tokenizer
        encoded_ids = [
            tokenizer.encode(str(action), add_special_tokens=False)
            for action in ACTION_NAMES
        ]
        if any(len(token_ids) != 1 for token_ids in encoded_ids):
            raise ValueError("Qwen action IDs 0-9 must each be a single tokenizer token")
        self.action_token_ids = torch.tensor(
            [token_ids[0] for token_ids in encoded_ids],
            device=config.device,
        )
        answer_ids = [
            tokenizer.encode(answer, add_special_tokens=False)
            for answer in ("YES", "NO")
        ]
        if any(len(token_ids) != 1 for token_ids in answer_ids):
            raise ValueError("Qwen YES and NO answers must each be one tokenizer token")
        self.answer_token_ids = torch.tensor(
            [token_ids[0] for token_ids in answer_ids],
            device=config.device,
        )

    def _prepare(self, messages):
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.config.device)

    def _next_token_probabilities(self, inputs, token_ids):
        with torch.inference_mode():
            # Policy and binary Value only need the final-token distribution.
            # Avoid materializing vocabulary logits for every image/text token.
            logits = self.model(
                **inputs,
                logits_to_keep=1,
                return_dict=True,
            ).logits[0, -1]
        return torch.softmax(logits[token_ids].float(), dim=0).cpu().tolist()

    def policy(
        self,
        instruction: str,
        images: Sequence,
        actions: Sequence[int],
        allow_stop: bool = True,
    ) -> PolicyOutput:
        images = [_as_pil(image) for image in images]
        content = []
        for index, image in enumerate(images):
            state = "current" if index == len(images) - 1 else "previous"
            content.extend(
                [
                    {"type": "text", "text": f"Image {index + 1}: {state} observation."},
                    {"type": "image", "image": image},
                ]
            )

        action_names = [ACTION_NAMES[action] for action in actions]
        available_actions = {
            action: name for action, name in ACTION_NAMES.items()
            if allow_stop or action != 0
        }
        action_space = ", ".join(
            f"{action}:{name}" for action, name in available_actions.items()
        )
        stop_instruction = (
            "Use STOP only after reaching the final target and when the drone appears to be within roughly 20 meters of it. Do not stop when the target is merely visible in the distance."
            if allow_stop
            else "STOP is handled separately and is not available in this decision."
        )
        content.append(
            {
                "type": "text",
                "text": f"""
You control a drone in a city visual-language navigation task.

Navigation instruction:
{instruction}

Executed actions: {json.dumps(action_names)}
Action space: {action_space}

Choose the single best next action.
{stop_instruction} prefer FORWARD_9 over shorter forward actions. Respond with exactly one action
ID digit from 0 to 9.
""".strip(),
            }
        )

        inputs = self._prepare([{"role": "user", "content": content}])
        probabilities = self._next_token_probabilities(inputs, self.action_token_ids)
        return PolicyOutput(tuple(probabilities))

    def evaluate_rollout(
        self,
        instruction: str,
        real_image_history: Sequence,
        leaf_image,
        action_history: Sequence[int],
        action_prefix: Sequence[int],
    ) -> RolloutEvaluation:
        image_items = [
            (f"Real navigation memory image {index}.", image)
            for index, image in enumerate(real_image_history, start=1)
        ]
        leaf_image_number = len(image_items) + 1
        image_items.append(
            ("Virtual leaf view after the full candidate rollout.", leaf_image)
        )

        content = []
        for image_number, (label, image) in enumerate(image_items, start=1):
            content.extend(
                [
                    {"type": "text", "text": f"Image {image_number}: {label}"},
                    {"type": "image", "image": _as_pil(image)},
                ]
            )

        history_names = [ACTION_NAMES[action] for action in action_history]
        prefix_names = [ACTION_NAMES[action] for action in action_prefix]
        question = f"""
After executing the candidate sequence and reaching Image {leaf_image_number}, is the
drone in a state from which it is likely to safely complete the navigation instruction
and reach the final target? Judge absolute task-success likelihood, not whether the
candidate is merely better than the current state. Answer NO for irrelevant, repetitive,
misaligned, overshooting, or dangerous movement.
""".strip()

        content.append(
            {
                "type": "text",
                "text": f"""
Judge a short candidate rollout for drone navigation.

Navigation instruction:
{instruction}

Real executed actions: {json.dumps(history_names)}
Candidate action sequence: {json.dumps(prefix_names)}

The images are chronological real navigation memory followed by one virtual leaf view.

{question}

Respond with exactly one word: YES or NO.
""".strip(),
            }
        )

        inputs = self._prepare([{"role": "user", "content": content}])
        yes_probability, no_probability = self._next_token_probabilities(
            inputs,
            self.answer_token_ids,
        )
        return RolloutEvaluation(
            success_probability=yes_probability,
            failure_probability=no_probability,
            question_type="task_success",
            real_image_count=len(real_image_history),
            virtual_image_count=1,
        )
