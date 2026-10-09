#!/usr/bin/env python3
"""The v4 question set (src/evaluation/datasets/questions.jsonl) against a deployed stack: every
question through POST /ask, scored like the offline evaluation (src/evaluation/answer.ts):
correctness against the oracle's answers, abstention, required evidence, retrieval metrics for the
semantic questions, the model's grounding when one answered, latency, tokens and cost — per
category, language and mode.

    python3 scripts/eval_v4.py https://pixagram-search-v4.<you>.workers.dev [--modes fast,auto] [--limit 400]
                               [--only temporal] [--out results.json] [--admin-token …] [--delay 0.5]

Image questions upload the artwork's image (fetched from the stack's own /posts and /img routes),
the same picture at twice its size (needs Pillow), or a generated one. Requests are paced
(--delay): the public /ask budget is 20 a minute per client, the admin token lifts it. Modes that
reason (balanced, deep, expert) spend Workers AI tokens: the cost is reported.
"""
from __future__ import annotations

import argparse
import base64
import collections
import io
import json
import math
import random
import statistics
import sys
import time
import urllib.error
import urllib.request

ABSTAIN = {"clarify", "not_found", "no_match", "insufficient_evidence"}


def http(method: str, url: str, body: dict | None = None, token: str | None = None, timeout: int = 120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"content-type": "application/json", "accept": "application/json"})
    if token:
        req.add_header("authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw[:1] in (b"{", b"[") else raw)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"error": raw[:200].decode("utf8", "replace")}


# ---- images ---------------------------------------------------------------------------------------

def novel_png(seed: int) -> bytes:
    """Pixel art no artwork shows (needs Pillow)."""
    from PIL import Image  # noqa: PLC0415

    rnd = random.Random(seed)
    pal = [tuple(rnd.randrange(256) for _ in range(3)) for _ in range(4)]
    im = Image.new("RGB", (24, 24))
    im.putdata([pal[rnd.randrange(4)] for _ in range(24 * 24)])
    out = io.BytesIO()
    im.save(out, "PNG")
    return out.getvalue()


def image_for(base: str, item: dict) -> bytes | None:
    im = item["image"]
    if im["transform"].startswith("novel:"):
        return novel_png(int(im["transform"][6:]))
    author, permlink = im["ref"][2:].split("/", 1)
    st, post = http("GET", f"{base}/posts/{author}/{permlink}")
    url = ((post or {}).get("artwork") or {}).get("images", {}).get("original") if st == 200 and isinstance(post, dict) else None
    if not url:
        return None
    with urllib.request.urlopen(url if url.startswith("http") else base + url, timeout=60) as r:
        raw = r.read()
    if im["transform"] == "scale2":
        from PIL import Image  # noqa: PLC0415

        img = Image.open(io.BytesIO(raw)).convert("RGBA")
        img = img.resize((img.width * 2, img.height * 2), Image.NEAREST)
        out = io.BytesIO()
        img.save(out, "PNG")
        raw = out.getvalue()
    return raw


# ---- scoring (mirrors src/evaluation/answer.ts) ------------------------------------------------------

def norm(v):
    return v.strip().lstrip("@").lower() if isinstance(v, str) else json.dumps(v)


def as_path(v):
    if not isinstance(v, str):
        return None
    s = v.strip()
    if s.startswith("/@"):
        return s
    if "/" in s:
        a, p = s.lstrip("@").split("/", 1)
        return f"/@{a}/{p}"
    return None


def matches(t, got, want):
    if want is None:
        return got is None
    if t == "author":
        return isinstance(got, str) and norm(got) == norm(want)
    if t == "post":
        return as_path(got) is not None and as_path(got) == as_path(want)
    if t == "date":
        return isinstance(got, str) and got[:10] == str(want)
    if t in ("count", "duration"):
        return isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - want) < 1e-9
    if t == "boolean":
        return got is want if isinstance(got, bool) else (isinstance(got, str) and isinstance(want, str) and norm(got) == norm(want))
    if t == "value":
        return norm(got) == norm(want)
    return False


def rank_metrics(ranked, rel):
    gains = [rel.get(r, 0) for r in ranked]
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal)) or 1
    nrel = len([v for v in rel.values() if v > 0]) or 1
    first = next((i for i, g in enumerate(gains) if g > 0), -1)
    hits, ap = 0, 0.0
    for i, g in enumerate(gains):
        if g > 0:
            hits += 1
            ap += hits / (i + 1)
    return {"ndcg10": dcg / idcg, "p10": len([g for g in gains[:10] if g > 0]) / 10, "r10": len([g for g in gains[:10] if g > 0]) / nrel, "mrr": 0 if first < 0 else 1 / (first + 1), "ap": ap / nrel}


def score(item, r):
    status = r.get("status")
    should = item.get("expected_status") in ABSTAIN
    abstained = status in ABSTAIN or r.get("answer") is None
    out = {"id": item["id"], "type": item["query_type"], "lang": item["lang"], "mode": r.get("mode"), "status": status, "got": r.get("answer")}
    t = item["answer_type"]
    if t == "status":
        ok = status == item.get("expected_status")
    elif t == "retrieval":
        order = list(dict.fromkeys([e.get("path") or f"/@{e['author']}/{e['permlink']}" for e in r.get("evidence") or []] + [f"/@{i['author']}/{i['permlink']}" for i in r.get("items") or []]))
        out["retrieval"] = rank_metrics(order, item.get("rel") or {})
        ok = bool(order) and (item.get("rel") or {}).get(order[0], 0) > 0
    else:
        wants = [item["expected_answer"], *item.get("acceptable_answers", [])]
        ok = any(matches(t, r.get("answer"), w) or (t == "boolean" and matches("author", r.get("answer"), w)) for w in wants)
        if item.get("expected_status") and status != item["expected_status"]:
            ok = False
    text = str(r.get("answer_text") or "").lower()
    if any(f.lower() in text for f in item.get("forbidden") or []):
        ok = False
    have = {e.get("path") or f"/@{e.get('author')}/{e.get('permlink')}" for e in r.get("evidence") or []}
    have |= {c.get("path") for c in r.get("cards") or [] if c.get("path")}
    have |= {f"/@{i['author']}/{i['permlink']}" for i in r.get("items") or []}
    req = item.get("required_evidence") or []
    out.update({
        "correct": ok, "should_abstain": should, "abstained": abstained, "abstention_ok": should == abstained,
        "evidence_ok": (all(p in have for p in req) if req and t != "status" else None),
        "egs": (r.get("grounding") or {}).get("egs"), "citation": (r.get("grounding") or {}).get("citation_accuracy"),
        "confidence": r.get("confidence"), "took_ms": r.get("took_ms") or 0,
        "model_ms": (r.get("usage") or {}).get("model_ms") or 0, "input_tokens": (r.get("usage") or {}).get("input_tokens") or 0,
        "output_tokens": (r.get("usage") or {}).get("output_tokens") or 0, "cost": (r.get("usage") or {}).get("cost_usd") or 0, "model": r.get("model"),
    })
    return out


def pct(xs, p):
    if not xs:
        return 0
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))]


def summary(rows):
    n = len(rows) or 1
    acc = sum(r["correct"] for r in rows) / n
    abst = sum(r["abstention_ok"] for r in rows) / n
    ev = [r["evidence_ok"] for r in rows if r["evidence_ok"] is not None]
    evacc = sum(ev) / len(ev) if ev else None
    grounding = statistics.fmean([r["egs"] if r["egs"] is not None else (1 if r["correct"] else 0) for r in rows]) if rows else 0
    rel = 0.4 * acc + 0.3 * grounding + 0.2 * (evacc if evacc is not None else acc) + 0.1 * abst
    eng = [max(0, r["took_ms"] - r["model_ms"]) for r in rows]
    ret = [r["retrieval"] for r in rows if "retrieval" in r]
    out = {
        "n": len(rows), "accuracy": acc, "abstention_accuracy": abst, "evidence_accuracy": evacc, "reliability": rel,
        "egs": statistics.fmean([r["egs"] for r in rows if r["egs"] is not None]) if any(r["egs"] is not None for r in rows) else None,
        "engine_ms": {"p50": pct(eng, 50), "p95": pct(eng, 95), "p99": pct(eng, 99)},
        "model_ms": {"p50": pct([r["model_ms"] for r in rows if r["model_ms"]], 50), "p95": pct([r["model_ms"] for r in rows if r["model_ms"]], 95)},
        "tokens": {"input": sum(r["input_tokens"] for r in rows), "output": sum(r["output_tokens"] for r in rows)},
        "cost_usd": sum(r["cost"] or 0 for r in rows),
        "confident_errors": sum(1 for r in rows if not r["correct"] and not r["abstained"] and (r["confidence"] or 0) >= 0.8),
    }
    if ret:
        out["retrieval"] = {k: statistics.fmean(x[k] for x in ret) for k in ret[0]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--dataset", default="src/evaluation/datasets/questions.jsonl")
    ap.add_argument("--modes", default="fast")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--admin-token", default="")
    ap.add_argument("--delay", type=float, default=3.2, help="seconds between requests (20/min without the admin token)")
    a = ap.parse_args()
    base = a.base.rstrip("/")
    items = [json.loads(l) for l in open(a.dataset, encoding="utf-8") if l.strip()]
    if a.only:
        items = [x for x in items if x["query_type"] == a.only]
    if a.limit:
        rnd = random.Random(1)
        rnd.shuffle(items)
        items = items[: a.limit]
    delay = 0.0 if a.admin_token else a.delay
    report = {}
    for mode in a.modes.split(","):
        rows = []
        for i, item in enumerate(items):
            body = {"question": item["question"], **({} if mode == "auto" else {"mode": mode})}
            if item.get("image"):
                if mode == "v3":
                    continue
                try:
                    img = image_for(base, item)
                except Exception as e:  # noqa: BLE001
                    print(f"  {item['id']}: image unavailable ({e})", file=sys.stderr)
                    continue
                if not img:
                    continue
                body["image"] = base64.b64encode(img).decode()
            t0 = time.time()
            st, r = http("POST", f"{base}/ask", body, a.admin_token or None)
            if st != 200 or not isinstance(r, dict):
                r = {"status": f"http_{st}", "answer": None, "answer_text": json.dumps(r)[:200], "took_ms": int((time.time() - t0) * 1000)}
            if mode == "v3":
                r = {**r, "status": "no_match" if r.get("answer") is None else "answered", "mode": "v3"}
            rows.append(score(item, r))
            if (i + 1) % 50 == 0:
                print(f"  {mode}: {i + 1}/{len(items)}", file=sys.stderr)
            time.sleep(delay)
        by_type = collections.defaultdict(list)
        by_lang = collections.defaultdict(list)
        for r in rows:
            by_type[r["type"]].append(r)
            by_lang[r["lang"]].append(r)
        report[mode] = {"all": summary(rows), "by_type": {k: summary(v) for k, v in by_type.items()}, "by_lang": {k: summary(v) for k, v in by_lang.items()}, "misses": [r for r in rows if not r["correct"]][:300]}
    cats = ["factual", "semantic", "visual", "temporal", "comparative", "multi_hop", "ambiguous", "multilingual", "adversarial"]
    print(f"{'category':13}" + "".join(f"{m:>22}" for m in report))
    for c in [*cats, "ALL"]:
        cells = []
        for m in report:
            g = report[m]["all"] if c == "ALL" else report[m]["by_type"].get(c)
            cells.append(f"{g['accuracy']:>10.3f} rel {g['reliability']:.2f} n{g['n']:<4}" if g else f"{'-':>22}")
        print(f"{c:13}" + "".join(cells))
    for m, r in report.items():
        s = r["all"]
        print(f"{m}: engine P50 {s['engine_ms']['p50']} ms P95 {s['engine_ms']['p95']} ms; model P50 {s['model_ms']['p50']} ms; tokens {s['tokens']}; cost ${s['cost_usd']:.4f}; EGS {s['egs']}; confident errors {s['confident_errors']}")
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
