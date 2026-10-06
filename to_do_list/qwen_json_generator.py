"""Minimal Qwen3-VL JSON generator for the standalone progress pipeline."""

from dataclasses import dataclass
import json
from pathlib import Path
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
MOVEMENT_ACTIONS = tuple(action for action in ACTION_NAMES if action != 0)


@dataclass(frozen=True)
class QwenGeneratorConfig:
    model_path: str
    device: str = "cuda:0"
    attention: str = "flash_attention_2"
    max_new_tokens: int = 512


@dataclass(frozen=True)
class ActionDecision:
    action: int
    probabilities: tuple[float, ...]

    @property
    def action_name(self):
        return ACTION_NAMES[self.action]

    def to_dict(self):
        return {
            "action": self.action,
            "action_name": self.action_name,
            "probabilities": {
                ACTION_NAMES[action]: probability
                for action, probability in zip(MOVEMENT_ACTIONS, self.probabilities)
            },
        }


def _as_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    return Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")


def extract_json_object(text: str) -> str:
    """Accept either raw JSON or a JSON object wrapped in a markdown fence."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Qwen response does not contain a JSON object: {text!r}")
    candidate = text[start : end + 1]
    json.loads(candidate)
    return candidate


class QwenJsonGenerator:
    def __init__(self, config: QwenGeneratorConfig):
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
        action_ids = [
            tokenizer.encode(str(action), add_special_tokens=False)
            for action in MOVEMENT_ACTIONS
        ]
        if any(len(token_ids) != 1 for token_ids in action_ids):
            raise ValueError("Qwen movement action IDs must be single tokens")
        self.action_token_ids = torch.tensor(
            [token_ids[0] for token_ids in action_ids],
            device=config.device,
        )

    def _prepare(self, prompt: str, images: Sequence):
        content = []
        for index, image in enumerate(images, start=1):
            state = "current" if index == len(images) else "previous"
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"Observation image {index}: {state} real view.",
                    },
                    {"type": "image", "image": _as_pil(image)},
                ]
            )
        content.append({"type": "text", "text": prompt})
        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.config.device)

    def __call__(self, prompt: str, images: Sequence) -> str:
        inputs = self._prepare(prompt, images)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )
        input_length = inputs["input_ids"].shape[1]
        text = self.processor.batch_decode(
            generated[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return extract_json_object(text)

    def choose_action(self, prompt: str, images: Sequence) -> ActionDecision:
        inputs = self._prepare(prompt, images)
        with torch.inference_mode():
            logits = self.model(
                **inputs,
                logits_to_keep=1,
                return_dict=True,
            ).logits[0, -1]
        probabilities = torch.softmax(
            logits[self.action_token_ids].float(),
            dim=0,
        ).cpu().tolist()
        best_index = int(np.argmax(probabilities))
        return ActionDecision(
            action=MOVEMENT_ACTIONS[best_index],
            probabilities=tuple(probabilities),
        )
