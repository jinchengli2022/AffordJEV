---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
library_name: transformers
tags:
- valen
- image-text-to-text
- decision-making
- decision-model
- image-text-to-typed-output
- structured-output
- shared-state
- qwen3.5
- custom-code
---

# Valen-4B

Valen-4B combines Qwen3.5-4B with a two-layer MLP-Mixer decision head. It predicts probability distributions for Choice, Noul (yes/no), and Score questions. A shared state can contain text, images, or video and serve multiple questions.

Valen-4B 基于 Qwen3.5-4B 和两层 MLP-Mixer 决策头，支持图文、文本、视频，以及共享 state 的多题推理。

## Training / 训练

- Two-stage SFT: decision-head warmup followed by joint LoRA, Mixer, and visual-merger fine-tuning. We train on millions of samples.
- Backbone: BF16. Mixer: FP32. Videos use 16 sampled frames.

## Files / 权重格式

- Root directory: complete backbone with merged LoRA, trained visual merger, FP32 Mixer, tokenizer, processor, and standalone loading code.

These weights include merged LoRA adapters. BF16 rounding can cause small probability differences from the original training checkpoints used for the reported benchmarks.

## Load merged weights / 加载完整合并权重

The verified environment uses Python 3.10, PyTorch 2.6.0, Transformers 5.4.0, torchvision 0.21.0, Pillow, and av 16.1.0. Flash Attention 2 requires a compatible flash-attn installation.

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "Valen-Team/Valen-4B",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",  # or "flash_attention_2"
).to("cuda").eval()
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.allow_tf32 = False

response = model.predict({
    "state": "There is one cat in the room.",
    "questions": {
        "animal": {"type": "choice", "instructions": "Which animal is present?",
                   "criteria": {"cat": "A cat", "dog": "A dog"}},
        "is_cat": {"type": "noul", "instructions": "There is a cat in the room."},
    },
}, execution="shared_state")
print(response["answers"])
```

For a local copy, replace the repository ID with `"./Valen-4B"`.
`execution="question"` scores questions separately. Images and video use the Valen `messages`, `image_url`, and `video_url` schema; relative media paths are resolved against `media_root`.
