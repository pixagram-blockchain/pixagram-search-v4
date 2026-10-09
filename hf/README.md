---
title: pixagram-search embeddings
emoji: 🟪
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: SigLIP image/text embeddings for the Pixagram search engine
---

# SigLIP embedding Space for pixagram-search

One model, two towers. The Cloudflare Worker sends artwork PNGs here while indexing and the
query text here at search time; both land in the same vector space, which is what makes
text → image search work. `app.py` is the Space entry point; it serves:

| route | |
|---|---|
| `POST /embed` | `{"inputs": {"images": ["<base64>", …], "texts": ["…", …]}}` → `{"model", "dim", "embeddings": [[…], …], "calibration", "max_num_patches"}` (images first, then texts; L2-normalised; `max_num_patches` for NaFlex models) |
| `GET /health` | `{"ok", "model", "dim", "ready", "stub", "calibration", "max_num_patches"}` — `ready` flips to true once the weights are loaded |
| `GET /` | Gradio page: embed a text or an image, score an image against captions |

`calibration` is the model's learned `{"logit_scale", "logit_bias"}`:
`sigmoid(logit_scale · cos + logit_bias)` is SigLIP's own text↔image match probability. The v3
Worker stores it (KV `calib:<model>`) and uses it until it has a background sample of the
corpus; older Spaces without the field still work (the Worker knows the SigLIP 2 base values).

## Deploy as a Space

1. Create a Space (SDK: **Gradio**, hardware: **CPU basic** is enough — a SigLIP-base pass on
   one artwork takes ~0.25 s on 2 vCPU, ~0.5 s for NaFlex at 576 patches). The Space computes
   one embedding per CPU at once and queues the rest; the Worker reads that number from
   `/health` (`concurrency`) every ten minutes and sends as many per consumer invocation, so
   bigger hardware is used with nothing else to change: on CPU upgrade (8 vCPU) it computes 8
   at once instead of 2.
2. Upload this folder — `README.md` (the frontmatter above is the Space config), `app.py`,
   `siglip.py`, `requirements.txt`; `handler.py` can stay, it is only used by Inference
   Endpoints. From the repository root:

   ```bash
   pip install -U huggingface_hub
   hf auth login                                                   # older CLI: huggingface-cli login
   hf upload <owner>/<space> hf . --repo-type space
   ```

   `scripts/deploy.sh` does all of this for the v4 Space (`primerz/pixagram-search-v4`); the v3
   repository did it for `primerz/pixagram-search-v3`.
3. Visibility:
   * **Private** Space (recommended): every request must carry an HF token that can read the
     Space — set that token as the Worker's `HF_TOKEN` secret. Leave `API_TOKEN` unset.
   * **Public** Space: set a Space secret `API_TOKEN` to a long random string and use the same
     string as the Worker's `HF_TOKEN`; `/embed` then rejects anything else with 401.
4. Worker: `npx wrangler secret put HF_TOKEN`, and set
   `HF_EMBED_URL = "https://<owner>-<space-name>.hf.space/embed"` in `wrangler.jsonc`
   (the host is `<owner>-<space-name>` with dots in the owner name replaced by dashes; the
   Space settings page shows the exact "Direct URL").
5. Deploy the Worker, then `scripts/admin.sh reindex-all embed,text` to fill the vectors in.

Cold start: the SigLIP base checkpoints are ~1.5 GB; the Space answers `/health`
in seconds and `/embed` blocks until the weights are loaded (about 20 s once cached, a few
minutes on the first build). Free CPU Spaces **sleep after 48 h without traffic**; the Worker
treats the wake-up page as a transient error and retries with backoff, and search simply runs
without the semantic leg until the Space is back. If that gap matters, use paid CPU hardware
and set the sleep time to "never".

## Concurrency

The Space computes `MAX_CONCURRENCY` embeddings at once (default: one per CPU), each with
`CPUs / MAX_CONCURRENCY` PyTorch threads (`TORCH_THREADS` overrides), on threads that live as long
as the server. Search queries (texts only) have threads of their own and never wait behind
images being indexed. `/health` answers at once even under load.

Measured on 2 vCPU (Oct 2026): 24 NaFlex images at 576 patches from 4 clients, alone, then with
a search text every second:

| | images alone | images with searches | a search text meanwhile | `/health` meanwhile |
|---|---|---|---|---|
| before: one request at a time, 2 threads | 1.81 / s | 1.76 / s | median 1.25 s, worst 1.9 s | blocked (seconds) |
| **one per CPU, 1 thread each** | 1.89 / s | 1.77 / s | median 0.22 s, worst 0.29 s | 3 ms |
| one at a time, 2 threads | 1.55 / s | | | |

Two images at once with one thread each beat one image with two threads (1.89 against 1.55
images/s here), which is why the default is one per CPU. Fewer at once with more threads each
(`MAX_CONCURRENCY=1`) answers a single query sooner (~70 ms instead of ~125 ms for a text on 2
vCPU) at the cost of throughput.

`python3 hf/loadtest.py https://<owner>-<space>.hf.space <token>` measures a running Space the
same way (the token is the Worker's `HF_TOKEN`, `SPACE_API_TOKEN` in `~/.pixagram-search-v4.json`):
run it before and after changing the hardware.

## Troubleshooting

* Text embeddings taking 8-10 s on a CPU Space: `os.cpu_count()` inside the container reports
  the host's cores (64-96) while the cgroup allows 2-8, and PyTorch spin-waits on the
  oversubscribed thread pool. `siglip.py` sizes everything from the cgroup quota
  (`effective_cpus()`); `/health` reports `cpus`, `concurrency` (embeddings at once) and
  `threads` (PyTorch threads per embedding) so you can see what it picked.

* `[Errno 98] error while attempting to bind on address ('0.0.0.0', 7860)` in the build/run
  log: Spaces set `GRADIO_SSR_MODE=True`, which makes Gradio start a Node SSR server on 7860
  before uvicorn. `app.py` mounts Gradio with `ssr_mode=False` for that reason — keep it.
* `WARNING: Running pip as the 'root' user…` during the build is harmless; the Space image
  installs requirements as root by design.
* `Your space is in error` on the direct URL: open the Space page → *Logs* (Build, then
  Container); `/health` only answers once the container is running.
* Intermittent `502` HTML pages (fast, ~0.15 s, no `x-proxied-host` / `x-proxied-replica`
  response header) while the Space shows *Running*: Hugging Face's edge fails before the
  request reaches the container, so the Container log shows nothing. On 2026-09-28 this came in
  windows of 10-50 s (up to 34 failures in a row). A quick retry does not bridge that. Search
  falls back to full text, and the queue retries with backoff. Restart the Space. If it
  persists, report it to HF with the `x-request-id` of a failed call, or move to paid hardware
  or an Inference Endpoint (`handler.py`).

## Test

```bash
curl -s https://<owner>-<space>.hf.space/health
curl -s https://<owner>-<space>.hf.space/embed -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" -d '{"inputs":{"texts":["a swan on a lake at sunset"]}}' | jq '.dim'
```

Local run (any machine with Python 3.10+): `pip install -r requirements.txt && python app.py`,
then point the Worker's `.dev.vars` at `HF_EMBED_URL=http://127.0.0.1:7860/embed`.
`EMBED_STUB=1 python app.py` starts without weights and returns deterministic pseudo-vectors —
handy for wiring tests, never for production (`/health` reports `"stub": true`).

## Changing the model

The code default is `google/siglip-base-patch16-256-multilingual`, which is what the
production Space runs. Choose a model **per Space** with the `MODEL_ID` variable, so that
uploading `hf/` never changes production's model:

- the v2 Space (`primerz/pixagram-siglip2`) sets `MODEL_ID=google/siglip2-base-patch16-256`;
- the v3 Space (`primerz/pixagram-search-v3`) and the v4 Space (`primerz/pixagram-search-v4`,
  created by this repository's `scripts/deploy.sh`) set `MODEL_ID=google/siglip2-base-patch16-naflex`
  and `MAX_NUM_PATCHES=576`: v4 embeds exactly as v3 does.

### NaFlex

NaFlex checkpoints keep each image's aspect ratio. The processor resizes the image to the
largest size that fits `MAX_NUM_PATCHES` patches of 16×16 pixels (default 256), and batches mix
shapes through an attention mask. Every reply reports `max_num_patches`. The Worker refuses
image vectors whose budget differs from its `EMBED_PATCHES` (text vectors do not depend on it),
and re-embeds when `EMBED_PATCHES` changes. On the Pixagram corpus, 256 patches did worse than
the fixed 256 px model and 576 slightly better (README-V3.md, "SigLIP 2 NaFlex").

Any SigLIP / SigLIP 2 checkpoint works unchanged: fixed-resolution ones load as `SiglipModel`,
NaFlex ones as `Siglip2Model`. Stick to **multilingual** checkpoints, because queries arrive
in any language: every SigLIP 2 model, `google/siglip-base-patch16-256-multilingual` or
`google/siglip-so400m-patch16-256-i18n`. The other SigLIP 1 checkpoints have an English text
tower.

Measured on the 106 Pixagram artworks (+500 pixel-art distractors, 157 queries in EN/FR/DE/JA,
2 vCPU, Sept 2026):

| model | dim | ms / image | ms / query | notes |
|---|---|---|---|---|
| siglip-base-patch16-256-multilingual | 768 | 310 | 90 | production |
| **siglip2-base-patch16-256** | 768 | 315 | 87 | v2 Space; same cost; EN/FR on par, DE/JA better |
| siglip2-base-patch16-384 / -naflex (256 patches) | 768 | 650 / 305 | 85 | no gain over base-256 |
| **siglip2-base-patch16-naflex, 576 patches** | 768 | ~530 | 85 | v3 Space: small gain, measured in Oct 2026 on 133 artworks (README-V3.md) |
| siglip2-large-patch16-256 | 1024 | 980 | 290 | no gain |
| siglip2-so400m-patch16-256 | 1152 | 1270 | 390 | small gain (JA) at 4× the CPU |
| siglip-so400m-patch16-256-i18n | 1152 | 1280 | 395 | best measured, 4× the CPU |

Switching a stack in place, same dimension (768 → 768): keep the Vectorize index. Set the
Space's `MODEL_ID`, then set `EMBED_MODEL` to the same id in that stack's wrangler config
and deploy. The Worker refuses vectors whose `model` differs from `EMBED_MODEL`, and
re-embeds artworks whose `embed_model` differs (the 10-minute sweeper picks them up; to do it
at once: `scripts/admin.sh reindex-all embed,text`). Then refresh the background sample
(`scripts/admin.sh background`), since z-scores are per model.

A different dimension also needs **new Vectorize indexes** (with the metadata indexes from
`scripts/lib.sh`) that `VEC` and `VEC_TEXT` point at, and a new `EMBED_DIM`. Vectors from
different models must never be compared.

## Inference Endpoint instead of a Space

`handler.py` implements the `EndpointHandler` contract for a dedicated Inference Endpoint
(task *custom*, same `siglip.py`, same request/response shape). Use it when you want
autoscaling, private networking, or GPU without the Space UI. Then `HF_EMBED_URL` is the
endpoint URL itself and the Worker's `X-Scale-Up-Timeout: 600` header makes a scaled-to-zero
replica wake up within the request instead of returning 503.
