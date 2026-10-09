#!/usr/bin/env python3
"""Local stand-in for the Workers AI cross-encoder: BAAI/bge-reranker-base (the model behind
@cf/baai/bge-reranker-base) behind a Text Embeddings Inference style POST /rerank, which the Worker
calls when SEARCH_RERANKER_MODEL=http and SEARCH_RERANK_URL points here (src/search/reranker.ts).
Used by the offline reranker evaluation (eval/offline/rerank.test.ts).

    HF_HOME=… python3 eval/offline/rerank_server.py --port 7862
    POST /rerank {"query": "…", "texts": ["…", …], "raw_scores": true} → [{"index": 0, "score": 1.23}, …]

Scores are the model's logits (raw_scores), as Workers AI returns them; the Worker applies the
sigmoid. Pairs are cut at 512 tokens, as the model was trained.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = os.environ.get("RERANK_MODEL_ID", "BAAI/bge-reranker-base")
TOK = AutoTokenizer.from_pretrained(MODEL_ID)
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
LOCK = threading.Lock()


def score(query: str, texts: list[str], batch: int = 16) -> list[float]:
    out: list[float] = []
    with LOCK, torch.inference_mode():
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            enc = TOK([query] * len(chunk), chunk, padding=True, truncation=True, max_length=512, return_tensors="pt")
            out.extend(MODEL(**enc).logits.view(-1).float().tolist())
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _send(self, code: int, payload) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"ok": True, "model": MODEL_ID}) if self.path.startswith("/health") else self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/rerank"):
            return self._send(404, {"error": "not found"})
        try:
            p = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
            q, texts = str(p.get("query") or ""), [str(t) for t in p.get("texts") or []]
            s = score(q, texts) if q and texts else []
            self._send(200, [{"index": i, "score": v} for i, v in enumerate(s)])
        except Exception as e:  # noqa: BLE001
            self._send(422, {"detail": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7862)
    a = ap.parse_args()
    print(f"rerank server: {MODEL_ID} on :{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
