#!/usr/bin/env python3
"""Checks jina_embedder.merge_lora: JinaCLIP v2's text tower with its LoRA adapter folded into the
weights gives the same embeddings as the original tower (adapter applied on every forward), and
different ones from the tower without the adapter.

    /root/venv-jina/bin/python eval/offline/jina_check.py      (transformers<5; ~5 GB of RAM)
"""
import os
import sys

import torch
from transformers import AutoModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jina_embedder import merge_lora  # noqa: E402

TEXTS = ["a red dragon", "un chat noir qui dort", "Sonnenuntergang über dem Meer", "pixel art of a knight on a horse"]

torch.set_num_threads(max(1, len(os.sched_getaffinity(0))))
m = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
del m.vision_model  # the adapter is in the text tower only
m.text_model.float()


def enc() -> torch.Tensor:
    with torch.inference_mode():
        return m.encode_text(TEXTS, task="retrieval.query", normalize_embeddings=True, convert_to_tensor=True).float()


ref = enc()
loraid = m.text_model._default_loraid
m.text_model._default_loraid = None
plain = enc()
m.text_model._default_loraid = loraid
merge_lora(m.text_model)
merged = enc()


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum(-1).min())


print(f"merged vs adapter on every forward: min cosine {cos(merged, ref):.6f}")
print(f"no adapter vs adapter on every forward: min cosine {cos(plain, ref):.6f}")
assert cos(merged, ref) > 0.9999, "merge changed the embeddings"
assert cos(plain, ref) < cos(merged, ref), "the adapter makes no difference: the check proves nothing"
print("ok")
