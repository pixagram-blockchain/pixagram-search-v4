"""
SigLIP embedder shared by app.py (Space) and handler.py (Inference Endpoint).

One model, two towers: images for the indexing pipeline, texts for search queries — both land
in the same vector space, which is what makes text → image search work. Vectors are
L2-normalised so cosine similarity is a dot product (Vectorize metric: cosine).

Text is lowercased here, before tokenisation. SigLIP 2 was trained on lowercased text and its
Gemma tokenizer is case-sensitive: on Pixagram artworks, Title Case queries drop from 97 % to
3 % recall@1. The original multilingual SigLIP tokenizer lowercases on its own, so this is a
no-op for it. Doing it here covers every caller (search queries, blog résumés, the demo page).

NaFlex models (google/siglip2-*-naflex) keep each image's aspect ratio: the processor resizes it to
the largest size that fits MAX_NUM_PATCHES patches of 16x16 px, instead of squashing it to a fixed
square. A 274x183 artwork is seen as about 19x13 patches rather than distorted to 16x16.

Environment:
  MODEL_ID         Hub id of a SigLIP/CLIP-family model (default: multilingual SigLIP base, 768-d;
                   the v2 Space primerz/pixagram-siglip2 sets google/siglip2-base-patch16-256, the
                   v3 Space google/siglip2-base-patch16-naflex)
  MAX_NUM_PATCHES  NaFlex only: patch budget per image (default 256, the model's training default).
                   Reported in every reply so the Worker can check it matches EMBED_PATCHES.
  MAX_BATCH        images/texts per forward pass (default 32)
  MAX_CONCURRENCY  embeddings computed at once (default: one per CPU, effective_cpus()); each
                   gets CPUs / MAX_CONCURRENCY PyTorch threads. One per CPU gives the most
                   throughput (a small forward pass does not spread well over many threads);
                   fewer, with more threads each, answer a single query sooner.
  TORCH_THREADS    PyTorch threads per embedding, overriding CPUs / MAX_CONCURRENCY
  EMBED_STUB       "1" returns deterministic pseudo-vectors without loading any model — wiring tests only
"""

from __future__ import annotations

import base64
import hashlib
import io
import math
import os
import threading
from typing import Any, Dict, List, Optional

from PIL import Image

# Default = what the production Space runs. Change models per Space with the MODEL_ID variable,
# so pushing hf/ never switches production's model by accident.
MODEL_ID = os.environ.get("MODEL_ID", "google/siglip-base-patch16-256-multilingual")
MAX_BATCH = int(os.environ.get("MAX_BATCH", "32"))
MAX_NUM_PATCHES = int(os.environ.get("MAX_NUM_PATCHES", "256"))
MAX_TEXT_TOKENS = 64  # SigLIP text-tower context
STUB = os.environ.get("EMBED_STUB", "") == "1"
STUB_DIM = int(os.environ.get("EMBED_STUB_DIM", "768"))


def decode_image(b64: str) -> Image.Image:
    """base64 (optionally a data URI) → RGB PIL image, transparency composited over white."""
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    img.load()
    return to_rgb(img)


def to_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    return img.convert("RGB")


class Embedder:
    """Lazy-loading, thread-safe wrapper around a SigLIP model."""

    def __init__(self, model_id: str = MODEL_ID, max_num_patches: int = MAX_NUM_PATCHES) -> None:
        self.model_id = model_id
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self.device = "cpu"
        self.dim = STUB_DIM if STUB else 0
        self.stub = STUB
        # NaFlex: set on load when the image processor takes a patch budget
        self.naflex = False
        self.max_num_patches = max_num_patches
        # sigmoid(logit_scale * cos + logit_bias) is the model's own text/image match probability
        self.calibration: Optional[Dict[str, float]] = None

    # ---- loading --------------------------------------------------------------------------

    def load(self) -> "Embedder":
        if self.stub or self._model is not None:
            return self
        with self._lock:
            if self._model is not None:
                return self
            import torch
            from transformers import AutoModel, AutoProcessor

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            # os.cpu_count() reports the HOST's cores inside a container (often 64-96 on Spaces)
            # while the cgroup allows 2-8; oversubscribing PyTorch's thread pool that badly turns
            # an 80 ms text embedding into 8-12 s of spin-waiting. Size the pool to what the
            # container can actually run, shared by the embeddings computed at once.
            torch.set_num_threads(threads_per_job())
            model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self.naflex = hasattr(getattr(self._processor, "image_processor", None), "max_num_patches")
            cfg = model.config
            self.dim = int(getattr(cfg, "projection_dim", 0) or getattr(cfg.text_config, "projection_size", 0) or cfg.text_config.hidden_size)
            # SigLIP's learned temperature and bias (exp() of the stored log-scale). CLIP-family models
            # without a bias report none.
            try:
                scale = float(model.logit_scale.exp().item())
                bias = float(model.logit_bias.item()) if getattr(model, "logit_bias", None) is not None else None
                self.calibration = {"logit_scale": round(scale, 4), "logit_bias": round(bias, 4)} if bias is not None else None
            except Exception:
                self.calibration = None
            self._model = model
        # First inference pays one-off kernel/allocator warm-up (~2 s); do it here, not on a user query.
        try:
            self.embed_texts(["warm up"])
            self.embed_images([Image.new("RGB", (32, 32), (128, 128, 128))])
        except Exception:
            pass
        return self

    @property
    def ready(self) -> bool:
        return self.stub or self._model is not None

    # ---- embedding ------------------------------------------------------------------------

    def embed_images(self, images: List[Image.Image]) -> List[List[float]]:
        if self.stub:
            return [_stub_vector(hashlib.sha256(im.tobytes()).hexdigest()) for im in images]
        self.load()
        import torch

        out: List[List[float]] = []
        # NaFlex: images of any aspect ratio share a batch; the processor pads them to the patch
        # budget and passes the attention mask and each image's patch grid to the model.
        kw = {"max_num_patches": self.max_num_patches} if self.naflex else {}
        with torch.inference_mode():
            for i in range(0, len(images), MAX_BATCH):
                batch = [to_rgb(im) for im in images[i : i + MAX_BATCH]]
                inputs = self._processor(images=batch, return_tensors="pt", **kw).to(self.device)
                feats = self._model.get_image_features(**inputs)
                feats = _pooled(feats)
                feats = torch.nn.functional.normalize(feats, dim=-1)
                out.extend(feats.cpu().float().tolist())
        return out

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.stub:
            return [_stub_vector(hashlib.sha256(t.strip().lower().encode()).hexdigest()) for t in texts]
        self.load()
        import torch

        out: List[List[float]] = []
        with torch.inference_mode():
            for i in range(0, len(texts), MAX_BATCH):
                # Lowercase: SigLIP 2 was trained on lowercased text (see module docstring).
                batch = [t.lower() if t.strip() else " " for t in texts[i : i + MAX_BATCH]]
                # SigLIP was trained with padding="max_length"; keep it so query vectors match.
                inputs = self._processor(
                    text=batch, padding="max_length", truncation=True, max_length=MAX_TEXT_TOKENS, return_tensors="pt"
                ).to(self.device)
                feats = self._model.get_text_features(**inputs)
                feats = _pooled(feats)
                feats = torch.nn.functional.normalize(feats, dim=-1)
                out.extend(feats.cpu().float().tolist())
        return out

    # ---- request handling shared by the Space route and the Endpoint handler -----------------

    def handle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        {"inputs": {"images": [b64, ...], "texts": [str, ...]}}  (either key optional)
        → {"model", "dim", "embeddings": [image vectors..., text vectors...], "calibration": {"logit_scale", "logit_bias"}}
        """
        inputs = payload.get("inputs", payload) if isinstance(payload, dict) else payload
        if isinstance(inputs, str):
            inputs = {"texts": [inputs]}
        if not isinstance(inputs, dict):
            raise ValueError("body must be {\"inputs\": {\"images\": [...], \"texts\": [...]}}")
        images_b64 = inputs.get("images") or []
        texts = inputs.get("texts") or []
        if isinstance(images_b64, str):
            images_b64 = [images_b64]
        if isinstance(texts, str):
            texts = [texts]
        if not images_b64 and not texts:
            raise ValueError("provide inputs.images (base64) and/or inputs.texts")
        if len(images_b64) + len(texts) > 256:
            raise ValueError("at most 256 items per request")

        embeddings: List[List[float]] = []
        if images_b64:
            embeddings.extend(self.embed_images([decode_image(b) for b in images_b64]))
        if texts:
            embeddings.extend(self.embed_texts([str(t) for t in texts]))
        out: Dict[str, Any] = {"model": self.model_id if not self.stub else "stub", "dim": len(embeddings[0]) if embeddings else self.dim, "embeddings": embeddings}
        if self.calibration:
            out["calibration"] = self.calibration
        if self.naflex:
            out["max_num_patches"] = self.max_num_patches
        return out


def effective_cpus(cap: int = 16) -> int:
    """CPUs this process may really use: affinity mask ∩ cgroup quota, capped."""
    n = os.cpu_count() or 1
    try:
        n = min(n, len(os.sched_getaffinity(0)))
    except Exception:
        pass
    try:  # cgroup v2
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except Exception:
        try:  # cgroup v1
            quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if quota > 0:
                n = min(n, max(1, quota // period))
        except Exception:
            pass
    return max(1, min(n, cap))


def concurrency() -> int:
    """Embeddings computed at once: MAX_CONCURRENCY, else one per CPU."""
    env = os.environ.get("MAX_CONCURRENCY", "")
    return max(1, int(env)) if env.isdigit() else effective_cpus()


def threads_per_job() -> int:
    """PyTorch threads per embedding: TORCH_THREADS, else the CPUs shared by the embeddings at once."""
    env = os.environ.get("TORCH_THREADS", "")
    if env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, effective_cpus() // concurrency())


def _pooled(feats):
    """transformers ≥5 may return a ModelOutput; older versions return the tensor directly."""
    if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
        return feats.pooler_output
    if hasattr(feats, "last_hidden_state") and not hasattr(feats, "shape"):
        return feats.last_hidden_state
    return feats


def _stub_vector(seed_hex: str) -> List[float]:
    """Deterministic unit vector from a hash — lets the Worker ↔ Space wiring be tested without weights."""
    vals: List[float] = []
    h = seed_hex
    while len(vals) < STUB_DIM:
        h = hashlib.sha256(h.encode()).hexdigest()
        vals.extend(int(h[i : i + 2], 16) / 127.5 - 1.0 for i in range(0, 64, 2))
    vals = vals[:STUB_DIM]
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]


_shared: Optional[Embedder] = None


def shared() -> Embedder:
    global _shared
    if _shared is None:
        _shared = Embedder()
    return _shared
