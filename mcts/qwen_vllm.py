"""Use the shared VLN prompts with a local vLLM next-token scoring service."""

import base64
import io
import math

import requests

from qwen_vln import ACTION_NAMES, QwenVLNBackend


class VllmQwenVLNBackend(QwenVLNBackend):
    def __init__(self, config, base_url, model_name):
        self.config = config
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.session = requests.Session()
        self.session.trust_env = False
        models = self.session.get(self.base_url + "/models", timeout=30)
        models.raise_for_status()
        if model_name not in {item["id"] for item in models.json()["data"]}:
            raise ValueError(f"Service does not expose model {model_name}")
        self.action_token_ids = [self._token_id(str(i)) for i in ACTION_NAMES]
        self.answer_token_ids = [self._token_id(word) for word in ("YES", "NO")]

    def _token_id(self, text):
        response = self.session.post(
            self.base_url.removesuffix("/v1") + "/tokenize",
            json={"model": self.model_name, "prompt": text, "add_special_tokens": False},
            timeout=30,
        )
        response.raise_for_status()
        tokens = response.json()["tokens"]
        if len(tokens) != 1:
            raise ValueError(f"Label {text!r} must be a single token: {tokens}")
        return tokens[0]

    def _prepare(self, messages):
        converted = []
        for message in messages:
            content = []
            for item in message["content"]:
                if item["type"] == "image":
                    buffer = io.BytesIO()
                    item["image"].save(buffer, format="PNG")
                    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                    content.append({"type": "image_url", "image_url": {
                        "url": "data:image/png;base64," + encoded,
                    }})
                else:
                    content.append(item)
            converted.append({"role": message["role"], "content": content})
        return converted

    def _next_token_probabilities(self, inputs, token_ids):
        # Request every label explicitly, rather than trusting a top-K list.
        # Normalizing log-probabilities over labels equals softmax of their logits.
        response = self.session.post(
            self.base_url + "/chat/completions",
            json={
                "model": self.model_name, "messages": inputs, "max_tokens": 1,
                "temperature": 1.0, "top_p": 1.0, "top_k": -1,
                "logprobs": True, "logprob_token_ids": list(token_ids),
                "return_tokens_as_token_ids": True,
                "chat_template_kwargs": {"enable_thinking": False},
                "seed": 0,
            },
            timeout=180,
        )
        response.raise_for_status()
        position = response.json()["choices"][0]["logprobs"]["content"][0]
        entries = position["top_logprobs"] + [position]
        values = {int(item["token"].removeprefix("token_id:")): item["logprob"]
                  for item in entries}
        scores = [values[token_id] for token_id in token_ids]
        if not all(math.isfinite(score) for score in scores):
            raise ValueError(f"Non-finite label log-probabilities: {scores}")
        peak = max(scores)
        weights = [math.exp(score - peak) for score in scores]
        total = sum(weights)
        return [weight / total for weight in weights]
