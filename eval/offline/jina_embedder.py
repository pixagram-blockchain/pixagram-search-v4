"""JinaCLIP v2 behind the same interface as hf/siglip.Embedder, for the offline evaluation only.

    BACKEND=jina JINA_TEXT=query python3 eval/offline/embed_server.py --port 7861

jinaai/jina-clip-v2 is CC BY-NC 4.0 (no commercial use of the weights), 1024-d, 512x512 images.
Its remote code needs transformers 4.x (it fails on 5.x), so run the server from an environment
with transformers<5, einops and timm.

Its text tower always applies one LoRA adapter (retrieval.query), recomputing W + BA for every
layer on every forward; for the 250k-token embedding matrix that is 2 GB of fp32 per call. The
adapter is merged into the weights once here instead (same outputs, flat memory), checked
against the unmerged model by eval/offline/jina_check.py.

JINA_TEXT: "query" prefixes every text with the model's query instruction (what its model card
does for text-to-image queries), "plain" sends texts as they are. The Worker cannot say which
texts are queries. README-V3's "query instruction" row built the index with "plain", then reran
the queries on the same STATE with "query".
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Any, Dict, List, Optional

from PIL import Image

# The Space's own decoding (transparency over white), so both models see the same pixels.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hf"))
from siglip import decode_image, to_rgb  # noqa: E402  (imports no ML library at module level)

MODEL_ID = os.environ.get("MODEL_ID", "jinaai/jina-clip-v2")
MAX_BATCH = int(os.environ.get("MAX_BATCH", "4"))
JINA_TEXT = os.environ.get("JINA_TEXT", "query")


def merge_lora(text) -> None:
    """Fold the text tower's default LoRA adapter into its weights (fp32), then switch the adapter off."""
    import torch
    from torch.nn.utils import parametrize

    loraid = text.default_loraid
    if loraid is None:
        return
    mods = [m for m in text.modules() if parametrize.is_parametrized(m, "weight")]
    mods.sort(key=lambda m: -m.parametrizations.weight.original.numel())  # the embedding first, while the rest is bf16
    with torch.no_grad():
        for m in mods:
            p = m.parametrizations.weight[0]
            orig = m.parametrizations.weight.original
            a, b = p.lora_A[loraid].float(), p.lora_B[loraid].float()
            merged = torch.matmul(*p.swap((b, a))).view(orig.shape)
            merged.mul_(p.scaling).add_(orig)
            # Keep the parametrization (its forward is the identity, and the attention code checks
            # for it to pick the LoRA layer's calling convention); swap the weights.
            orig.data = merged
            del merged
    text._default_loraid = None  # merged: no adapter_mask, so no second LoRA on top


class JinaEmbedder:
    naflex = False
    max_num_patches = None
    calibration: Optional[Dict[str, float]] = None  # CLIP-style: a temperature, no bias

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self.dim = 1024
        self._model = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> "JinaEmbedder":
        import torch
        from transformers import AutoModel

        torch.set_num_threads(max(1, len(os.sched_getaffinity(0))))
        model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
        merge_lora(model.text_model)
        model = model.float()
        self._model = model
        self.embed_texts(["warm up"])
        self.embed_images([Image.new("RGB", (32, 32), (128, 128, 128))])
        return self

    def embed_images(self, images: List[Image.Image]) -> List[List[float]]:
        import torch

        with self._lock, torch.inference_mode():
            v = self._model.encode_image([to_rgb(im) for im in images], batch_size=MAX_BATCH, normalize_embeddings=True)
        return [[float(x) for x in row] for row in v]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        import torch

        task = "retrieval.query" if JINA_TEXT == "query" else None
        with self._lock, torch.inference_mode():
            v = self._model.encode_text([t if t.strip() else " " for t in texts], task=task, batch_size=32, normalize_embeddings=True)
        return [[float(x) for x in row] for row in v]

    def handle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        inputs = payload.get("inputs", payload) if isinstance(payload, dict) else payload
        if isinstance(inputs, str):
            inputs = {"texts": [inputs]}
        images_b64 = inputs.get("images") or []
        texts = inputs.get("texts") or []
        if isinstance(images_b64, str):
            images_b64 = [images_b64]
        if isinstance(texts, str):
            texts = [texts]
        if not images_b64 and not texts:
            raise ValueError("provide inputs.images (base64) and/or inputs.texts")
        out: List[List[float]] = []
        if images_b64:
            out.extend(self.embed_images([decode_image(b) for b in images_b64]))
        if texts:
            out.extend(self.embed_texts([str(t) for t in texts]))
        # No max_num_patches: the Worker only checks it when the reply carries one.
        return {"model": self.model_id, "dim": self.dim, "embeddings": out}
