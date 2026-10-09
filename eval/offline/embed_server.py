#!/usr/bin/env python3
"""Local stand-in for the Hugging Face Space: the same siglip.Embedder.handle() behind POST /embed,
on the standard library HTTP server (no Gradio). Used by the offline evaluation.

    MODEL_ID=google/siglip2-base-patch16-naflex MAX_NUM_PATCHES=256 python3 eval/offline/embed_server.py --port 7861
    BACKEND=jina JINA_TEXT=query python3 eval/offline/embed_server.py --port 7861   # JinaCLIP v2, see jina_embedder.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hf"))
if os.environ.get("BACKEND") == "jina":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from jina_embedder import JinaEmbedder  # noqa: E402

    EMB = JinaEmbedder().load()
else:
    os.environ.setdefault("MODEL_ID", "google/siglip2-base-patch16-naflex")
    import siglip  # noqa: E402

    EMB = siglip.Embedder(os.environ["MODEL_ID"]).load()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, {"ok": True, "model": EMB.model_id, "dim": EMB.dim, "ready": EMB.ready, "calibration": EMB.calibration,
                             "max_num_patches": EMB.max_num_patches if EMB.naflex else None})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/embed"):
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("content-length") or 0)
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
            self._send(200, EMB.handle(payload))
        except ValueError as e:
            self._send(400, {"detail": str(e)})
        except Exception as e:  # noqa: BLE001
            self._send(422, {"detail": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7861)
    a = ap.parse_args()
    print(f"embed server: {EMB.model_id} dim={EMB.dim} calibration={EMB.calibration} naflex={EMB.naflex} patches={EMB.max_num_patches} on :{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
