"""Reuse the loaded Qwen baseline instance for an additional grounding call."""


class QwenWaypointGenerator:
    def __init__(self, backend, max_new_tokens: int = 160):
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.backend = backend
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt, images):
        import torch
        from mcts.qwen_vln import _as_pil

        content = []
        for index, image in enumerate(images):
            state = "current" if index == len(images) - 1 else "previous"
            content.extend([
                {"type": "text", "text": f"Image {index + 1}: {state} observation."},
                {"type": "image", "image": _as_pil(image)},
            ])
        content.append({"type": "text", "text": prompt})
        inputs = self.backend._prepare([{"role": "user", "content": content}])
        with torch.inference_mode():
            generated = self.backend.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            )
        input_length = inputs["input_ids"].shape[1]
        return self.backend.processor.batch_decode(
            generated[:, input_length:], skip_special_tokens=True,
        )[0].strip()
