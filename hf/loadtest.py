#!/usr/bin/env python3
"""Throughput of a running Space: 24 images from 4 clients at once (NaFlex-sized, 480x480), with a
search text every second, then the same without the texts. Run it before and after changing the
Space's hardware or MAX_CONCURRENCY.

    python3 hf/loadtest.py https://primerz-pixagram-search-v4.hf.space [token]

The token is the Worker's HF_TOKEN (SPACE_API_TOKEN in ~/.pixagram-search-v4.json) when the
Space has an API_TOKEN; it is read from $SPACE_API_TOKEN when not given.
"""
import base64
import io
import json
import os
import sys
import threading
import time
import urllib.request

from PIL import Image

BASE = sys.argv[1].rstrip("/")
TOKEN = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("SPACE_API_TOKEN", "")
buf = io.BytesIO()
Image.effect_noise((480, 480), 64).convert("RGB").save(buf, "PNG")
IMG = base64.b64encode(buf.getvalue()).decode()


def post(body):
    headers = {"content-type": "application/json", **({"authorization": f"Bearer {TOKEN}"} if TOKEN else {})}
    req = urllib.request.Request(BASE + "/embed", data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def run(with_texts: bool) -> str:
    done, text_ms = False, []

    def client():
        for _ in range(6):
            post({"inputs": {"images": [IMG]}})

    def texts():
        while not done:
            time.sleep(1)
            a = time.time()
            post({"inputs": {"texts": ["a red dragon over a castle"]}})
            text_ms.append((time.time() - a) * 1000)

    t0 = time.time()
    clients = [threading.Thread(target=client) for _ in range(4)]
    tx = threading.Thread(target=texts) if with_texts else None
    for c in clients:
        c.start()
    if tx:
        tx.start()
    for c in clients:
        c.join()
    wall = time.time() - t0
    done = True
    if tx:
        tx.join()
    out = f"{24 / wall:.2f} images/s"
    if text_ms:
        text_ms.sort()
        out += f"; a search text meanwhile: median {text_ms[len(text_ms) // 2]:.0f} ms, worst {text_ms[-1]:.0f} ms"
    return out


if __name__ == "__main__":
    with urllib.request.urlopen(BASE + "/health", timeout=60) as r:
        h = json.load(r)
    print("health:", {k: h.get(k) for k in ("model", "cpus", "concurrency", "threads")})
    post({"inputs": {"texts": ["warm up"]}})
    post({"inputs": {"images": [IMG]}})
    print("images alone:       ", run(False))
    print("images and searches:", run(True))
