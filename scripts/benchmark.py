#!/usr/bin/env python3
"""Benchmark reasoning models on Pixagram's own questions (spec §31, §58): every model receives
exactly the same question, retrieval results, evidence and instructions.

For each question, POST /admin/ask/context freezes the context once (retrieval, operators and
evidence verification run; no model); then POST /admin/ask/reason gives that same context to each
model, and the Worker verifies the reply's claims against the same cards. Reported per model:
grounding (EGS), citation accuracy, answers rejected by claim verification, how often the answer
states the index's own result (what /ask requires before showing a model's explanation), agreement
with the question set's expected answer, abstention, reasoning/output tokens, cost, and latency.

    ADMIN_TOKEN=… python3 scripts/benchmark.py https://pixagram-search-v4.<you>.workers.dev \
        [--models @cf/openai/gpt-oss-120b,@cf/nvidia/nemotron-3-120b-a12b,@cf/google/gemma-4-26b-a4b-it]
        [--reasoning medium] [--limit 60] [--types temporal,comparative,multi_hop,semantic,factual] [--out bench.json]

Costs Workers AI tokens (about one context of 2–6k tokens per question and model). The default
models are the spec's §58 list; the question sample is drawn evenly from the given categories.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

DEFAULT_MODELS = [
    "@cf/openai/gpt-oss-120b",
    "@cf/nvidia/nemotron-3-120b-a12b",
    "@cf/google/gemma-4-26b-a4b-it",
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "@cf/moonshotai/kimi-k2.6",
]

YES = re.compile(r"^\s*(yes|oui|ja|s[ií])\b", re.I)
NO = re.compile(r"^\s*(no|non|nein)\b", re.I)


def post(url: str, body: dict, token: str, timeout: int = 180):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers={"content-type": "application/json", "authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except ValueError:
            return e.code, {"error": f"HTTP {e.code}"}


def agrees(item: dict, answer: str | None, ctx: dict) -> bool | None:
    """Does a model's written answer state the expected answer? None when the type cannot be read from text."""
    if not answer:
        return False
    t, want = item["answer_type"], item["expected_answer"]
    wants = [want, *item.get("acceptable_answers", [])]
    a = answer.lower()
    if t == "author":
        return any(isinstance(w, str) and re.search(rf"@?{re.escape(w.lower())}\b", a) for w in wants)
    if t == "date":
        return any(isinstance(w, str) and w in a for w in wants)
    if t in ("count", "duration"):
        nums = {float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", a)}
        return any(isinstance(w, (int, float)) and float(w) in nums for w in wants)
    if t == "boolean":
        return bool(YES.match(answer)) if want is True else bool(NO.match(answer)) if want is False else None
    if t == "post":
        cards = {c.get("path"): c.get("title") for c in ctx.get("cards", []) if c.get("path")}
        return any(isinstance(w, str) and (w.lower() in a or (cards.get(w) and cards[w].lower() in a)) for w in wants)
    if t == "status":
        return bool(re.search(r"insufficient|not enough|cannot determine|pas assez|nicht genug|insuficiente|insufficient", a)) if item.get("expected_status") in ("not_found", "no_match") else None
    return None


def pct(xs, p):
    if not xs:
        return 0
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--dataset", default="src/evaluation/datasets/questions.jsonl")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--reasoning", default="medium", choices=["none", "low", "medium", "high"])
    ap.add_argument("--max-output-tokens", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--types", default="factual,temporal,comparative,multi_hop,semantic,multilingual,adversarial")
    ap.add_argument("--mode", default="deep", choices=["balanced", "deep", "expert"])
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    token = os.environ.get("ADMIN_TOKEN") or ""
    if not token:
        print("set ADMIN_TOKEN (the stack's admin token)", file=sys.stderr)
        return 2
    base = a.base.rstrip("/")
    types = a.types.split(",")
    items = [json.loads(l) for l in open(a.dataset, encoding="utf-8") if l.strip()]
    items = [x for x in items if x["query_type"] in types and not x.get("image")]
    rnd = random.Random(a.seed)
    by = collections.defaultdict(list)
    for x in items:
        by[x["query_type"]].append(x)
    sample = []
    per = max(1, a.limit // max(1, len(by)))
    for t in types:
        rnd.shuffle(by[t])
        sample += by[t][:per]
    sample = sample[: a.limit]
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    print(f"{len(sample)} questions × {len(models)} models, reasoning {a.reasoning}, contexts frozen in mode {a.mode}", file=sys.stderr)

    contexts = []
    for x in sample:
        st, ctx = post(f"{base}/admin/ask/context", {"question": x["question"], "mode": a.mode}, token)
        if st != 200:
            print(f"  context failed for {x['id']}: {ctx}", file=sys.stderr)
            continue
        contexts.append((x, ctx))
    runs: dict[str, list] = {m: [] for m in models}
    for m in models:
        for i, (x, ctx) in enumerate(contexts):
            body = {"question": ctx["question"], "lang": ctx["lang"], "cards": ctx["cards"], "context": ctx["context"], "shown": ctx.get("shown", []), "model": m, "reasoning": a.reasoning, "max_output_tokens": a.max_output_tokens}
            t0 = time.time()
            st, r = post(f"{base}/admin/ask/reason", body, token)
            r = r if isinstance(r, dict) else {"error": str(r)}
            r["wall_ms"] = int((time.time() - t0) * 1000)
            r["item"] = x["id"]
            r["agrees"] = agrees(x, r.get("answer"), ctx) if not r.get("error") else False
            r["deterministic_agrees"] = None
            runs[m].append(r)
            if (i + 1) % 20 == 0:
                print(f"  {m}: {i + 1}/{len(contexts)}", file=sys.stderr)

    print(f"\n{'model':44}{'ok':>5}{'EGS':>7}{'cite':>7}{'rej':>6}{'states':>7}{'agree':>7}{'abst':>6}{'in tok':>9}{'out tok':>9}{'$/100q':>9}{'P50 ms':>8}{'P95 ms':>8}")
    table = {}
    for m, rs in runs.items():
        good = [r for r in rs if not r.get("error")]
        g = [r["grounding"] for r in good if r.get("grounding")]
        egs = statistics.fmean(x["egs"] for x in g) if g else 0
        cite = statistics.fmean(x["citation_accuracy"] for x in g) if g else 0
        rejected = sum(1 for x in g if x["answer"] in ("unsupported", "contradicted"))
        agree = [r["agrees"] for r in good if r["agrees"] is not None]
        # the Worker's own test: does the answer state the index's result (what /ask requires to show it)
        states = [r["agreement"]["ok"] for r in good if r.get("agreement")]
        abst = sum(1 for r in good if r.get("status") == "insufficient_evidence")
        tin = sum((r.get("usage") or {}).get("inputTokens", 0) for r in good)
        tout = sum((r.get("usage") or {}).get("outputTokens", 0) for r in good)
        cost = sum(r.get("cost_usd") or 0 for r in good)
        lat = [r.get("model_ms") or 0 for r in good]
        table[m] = {"ok": len(good), "errors": len(rs) - len(good), "egs": egs, "citation_accuracy": cite, "rejected": rejected, "states_result": (sum(states) / len(states)) if states else None, "agreement": (sum(agree) / len(agree)) if agree else None, "abstained": abst, "input_tokens": tin, "output_tokens": tout, "cost_usd": cost, "cost_per_100": 100 * cost / max(1, len(good)), "p50_ms": pct(lat, 50), "p95_ms": pct(lat, 95)}
        t = table[m]
        print(f"{m:44}{t['ok']:>5}{t['egs']:>7.3f}{t['citation_accuracy']:>7.3f}{t['rejected']:>6}{(t['states_result'] or 0):>7.3f}{(t['agreement'] or 0):>7.3f}{t['abstained']:>6}{t['input_tokens']:>9}{t['output_tokens']:>9}{t['cost_per_100']:>9.4f}{t['p50_ms']:>8}{t['p95_ms']:>8}")
    if a.out:
        json.dump({"models": table, "reasoning": a.reasoning, "questions": [x["id"] for x, _ in contexts], "runs": runs}, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
