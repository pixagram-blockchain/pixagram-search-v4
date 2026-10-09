#!/usr/bin/env python3
"""Learn ranker weights from what users did with search results (learning to rank).

    python3 scripts/train-ranker.py --base https://pixagram-search-v3.<you>.workers.dev --token $ADMIN_TOKEN
    python3 scripts/train-ranker.py --file export.json --out weights.json      # offline, from a saved export

Data: GET /admin/ltr/export (rank_log ⋈ feedback): every result shown for a relevance-ordered
query that got at least one reaction, with the features the ranker saw and a graded label
(0 shown only, 1 click/open, 2 dwell/similar, 3 like/save).

Clicks are biased towards the top of the page: an item at rank 8 that got clicked says more than
one at rank 1. Each label is weighted by the inverse of the examination propensity
(1/rank)^eta (inverse propensity scoring, eta = 1 by default, capped), and the objective is the
mean IPS-weighted nDCG@10 of the re-ranked lists. Weights are fitted by coordinate ascent from the
stack's active weights, with k-fold cross-validation over queries. The weights are only written
out when they beat the active ones on held-out queries (--force to override).

Install: POST /admin/ranker/weights with the output file (scripts/admin.sh weights-set weights.json);
revert: POST {"reset": true}. Fitting on editorial judgments instead is eval/offline/fit_weights.py.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import urllib.request
from collections import defaultdict

FEATURES = ["title", "lexical", "coverage", "tag", "author", "concept", "sem", "sem_txt", "color", "tone", "orientation"]
LEXICAL = {"title", "lexical", "coverage", "tag"}
COLOR = {"color", "tone"}
# what this script may move (the z-score shape of "sem" is not in the logs, only its value)
PARAMS = [("w", k) for k in FEATURES] + [("colorLedLexical", None), ("colorBoost", None), ("dupFactor", None)]


def get_json(url: str, token: str) -> dict:
    req = urllib.request.Request(url, headers={"authorization": f"Bearer {token}", "accept": "application/json", "user-agent": "pixagram-search-ltr/3.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def score(x: dict, p: dict) -> float:
    led = bool(x.get("color_led"))
    rel = 0.0
    for k in FEATURES:
        w = p["w"][k]
        if led:
            if k in LEXICAL:
                w *= p["colorLedLexical"]
            elif k in COLOR:
                w *= p["colorBoost"]
        rel += w * float(x.get(k, 0.0))
    qf = p["quality"][0] + (p["quality"][1] - p["quality"][0]) * float(x.get("quality", 0.5))
    ff = p["freshness"][0] + (p["freshness"][1] - p["freshness"][0]) * float(x.get("freshness", 0.5))
    s = rel * qf * ff
    if x.get("dup"):
        s *= p["dupFactor"]
    return s


def group(rows: list[dict], eta: float, cap: float) -> list[dict]:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["qid"]].append(r)
    qs = []
    for qid, items in by.items():
        if len(items) < 2 or not any(i["label"] > 0 for i in items):
            continue
        for i in items:
            # IPS: a reaction at rank r counts (r)^eta times as much as one at rank 1 (capped)
            i["w"] = min(cap, max(1.0, float(i.get("rank") or 1)) ** eta) if i["label"] > 0 else 0.0
        gains = sorted(((2 ** i["label"] - 1) * i["w"] for i in items), reverse=True)
        ideal = sum(g / math.log2(k + 2) for k, g in enumerate(gains[:10])) or 1.0
        qs.append({"qid": qid, "q": items[0].get("q", ""), "items": items, "ideal": ideal})
    return qs


def ndcg(q: dict, p: dict) -> float:
    ranked = sorted(q["items"], key=lambda i: (-score(i["features"], p), i["rank"]))
    dcg = sum((2 ** i["label"] - 1) * i["w"] / math.log2(k + 2) for k, i in enumerate(ranked[:10]))
    return dcg / q["ideal"]


def mean_ndcg(qs: list[dict], p: dict) -> float:
    return sum(ndcg(q, p) for q in qs) / max(1, len(qs))


def get(p: dict, key):
    a, b = key
    return p[a][b] if b else p[a]


def put(p: dict, key, v) -> None:
    a, b = key
    if b:
        p[a][b] = v
    else:
        p[a] = v


def candidates_for(key, v: float) -> list[float]:
    a, _ = key
    if a == "dupFactor":
        return [x for x in (0.8, 0.85, 0.9, 0.95, 1.0) if abs(x - v) > 1e-9]
    base = v if v > 0 else 0.05
    return [0.0] + [min(5.0, base * m) for m in (0.25, 0.5, 0.75, 1.33, 2.0, 4.0)]


def fit(qs: list[dict], start: dict, passes: int = 4, seed: int = 0) -> dict:
    rng = random.Random(seed)
    p = json.loads(json.dumps(start))
    best = mean_ndcg(qs, p)
    for _ in range(passes):
        improved = False
        order = PARAMS[:]
        rng.shuffle(order)
        for key in order:
            cur = get(p, key)
            for v in candidates_for(key, cur):
                put(p, key, v)
                s = mean_ndcg(qs, p)
                if s > best + 1e-6:
                    best, cur, improved = s, v, True
            put(p, key, cur)
        if not improved:
            break
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", help="stack URL (reads /admin/ltr/export and /admin/ranker/weights)")
    ap.add_argument("--token", help="ADMIN_TOKEN of that stack")
    ap.add_argument("--file", help="a saved /admin/ltr/export JSON instead of --base")
    ap.add_argument("--weights", help="starting weights JSON (default: the stack's active weights)")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--eta", type=float, default=1.0, help="position-bias exponent (0 = no correction)")
    ap.add_argument("--cap", type=float, default=10.0, help="largest IPS weight")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-queries", type=int, default=50)
    ap.add_argument("--out", default="weights.json")
    ap.add_argument("--force", action="store_true", help="write the weights even without a held-out gain")
    a = ap.parse_args()

    if a.file:
        rows = json.load(open(a.file))
    elif a.base and a.token:
        rows = get_json(f"{a.base.rstrip('/')}/admin/ltr/export?days={a.days}", a.token)
    else:
        ap.error("give --base and --token, or --file")
    if a.weights:
        start = json.load(open(a.weights))
    elif a.base and a.token:
        start = get_json(f"{a.base.rstrip('/')}/admin/ranker/weights", a.token)["active"]
    else:
        ap.error("give --weights when training from --file")
    for k in ("quality", "freshness", "colorLedLexical", "colorBoost", "dupFactor"):
        if k not in start:
            sys.exit(f"starting weights lack {k!r}: use the 'active' object of GET /admin/ranker/weights")

    qs = group(rows, a.eta, a.cap)
    print(f"{len(rows)} logged results, {len(qs)} queries with a reaction")
    if len(qs) < a.min_queries:
        print(f"not enough queries to learn from (< {a.min_queries}); keep collecting /feedback")
        return 1
    print(f"active weights: IPS-nDCG@10 {mean_ndcg(qs, start):.4f}")

    # k-fold cross-validation over queries
    rng = random.Random(7)
    order = list(range(len(qs)))
    rng.shuffle(order)
    held_fit, held_cur = [], []
    for f in range(a.folds):
        test = [qs[i] for j, i in enumerate(order) if j % a.folds == f]
        train = [qs[i] for j, i in enumerate(order) if j % a.folds != f]
        if not test or not train:
            continue
        p = fit(train, start, seed=f)
        held_fit.append(mean_ndcg(test, p))
        held_cur.append(mean_ndcg(test, start))
        print(f"  fold {f}: held-out fitted {held_fit[-1]:.4f}  active {held_cur[-1]:.4f}")
    gain = sum(held_fit) / len(held_fit) - sum(held_cur) / len(held_cur)
    print(f"held-out gain: {gain:+.4f}")

    final = fit(qs, start, passes=6)
    final["w"] = {k: round(v, 4) for k, v in final["w"].items()}
    print(f"fitted on all: IPS-nDCG@10 {mean_ndcg(qs, final):.4f}")
    print("weights:", json.dumps(final["w"]), f"colorLedLexical={final['colorLedLexical']:.3g} colorBoost={final['colorBoost']:.3g} dupFactor={final['dupFactor']:.3g}")
    if gain <= 0 and not a.force:
        print("no held-out gain: not writing weights (--force to write anyway)")
        return 1
    final["version"] = f"ltr-{time.strftime('%Y%m%d')}-q{len(qs)}"
    json.dump(final, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}; install with: scripts/admin.sh weights-set {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
