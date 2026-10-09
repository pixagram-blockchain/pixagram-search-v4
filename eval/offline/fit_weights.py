#!/usr/bin/env python3
"""Fit the ranker's weights on judged queries (coordinate ascent on nDCG@10), with cross-validation.

    python3 eval/offline/fit_weights.py candidates.jsonl            # CV report + fitted weights
    python3 eval/offline/fit_weights.py candidates.jsonl --out weights.json

Input: the candidate dump of eval/offline/dump.test.ts (features of every candidate of every
judged query, with the raw z-scores of the semantic legs). The same linear scorer as
src/search/ranker.ts is reproduced here; the fitted JSON can be installed with
POST /admin/ranker/weights or used as the new defaults. Fitting on clicks instead of editorial
judgments is scripts/train-ranker.py.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict

FEATURES = ["title", "lexical", "coverage", "tag", "author", "concept", "sem", "sem_txt", "color", "tone", "orientation"]
LEXICAL = {"title", "lexical", "coverage", "tag"}
COLOR = {"color", "tone"}

DEFAULT = {
    "w": {"title": 0.4, "lexical": 0.1, "coverage": 0.08, "tag": 0.1, "author": 0.25, "concept": 0.15, "sem": 0.45, "sem_txt": 0.1, "color": 0.2, "tone": 0.15, "orientation": 0.05},
    "semZ0": 2.5, "semSlope": 1.5, "txtZ0": 2.5, "txtSlope": 1.5,
    "colorLedLexical": 0.3, "colorBoost": 5,
    "quality": [0.9, 1.1], "freshness": [0.95, 1.05], "dupFactor": 0.92,
}


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x > -60 else 0.0


def load(path: str):
    qs: dict[int, dict] = {}
    for line in open(path):
        c = json.loads(line)
        q = qs.setdefault(c["qi"], {"q": c["q"], "cat": c["cat"], "cands": [], "ideal": None})
        q["cands"].append(c)
    for q in qs.values():
        rels = sorted((c["rel"] for c in q["cands"]), reverse=True)
        # ideal from the judgments themselves would need all judged items; candidates cover the corpus
        q["ideal"] = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(rels[:10])) or 1.0
    return [qs[k] for k in sorted(qs)]


def score(c: dict, p: dict) -> float:
    f = dict(c["f"])
    z = c.get("z") or {}
    if z.get("zImage") is not None:
        f["sem"] = sigmoid(p["semSlope"] * (z["zImage"] - p["semZ0"]))
    if z.get("zText") is not None:
        f["sem_txt"] = sigmoid(p["txtSlope"] * (z["zText"] - p["txtZ0"]))
    rel = 0.0
    for k in FEATURES:
        w = p["w"][k]
        if c["colorLed"]:
            if k in LEXICAL:
                w *= p["colorLedLexical"]
            elif k in COLOR:
                w *= p["colorBoost"]
        rel += w * f[k]
    qf = p["quality"][0] + (p["quality"][1] - p["quality"][0]) * c["quality"]
    ff = p["freshness"][0] + (p["freshness"][1] - p["freshness"][0]) * c["freshness"]
    s = rel * qf * ff
    if c.get("dup") is not None:
        s *= p["dupFactor"]
    return s


def ndcg(q: dict, p: dict) -> float:
    ranked = sorted(q["cands"], key=lambda c: -score(c, p))
    dcg = sum((2 ** c["rel"] - 1) / math.log2(i + 2) for i, c in enumerate(ranked[:10]))
    return dcg / q["ideal"]


def mean_ndcg(qs, p) -> float:
    return sum(ndcg(q, p) for q in qs) / len(qs)


# parameters the search may move, with how
PARAMS = [("w", k) for k in FEATURES] + [("semZ0", None), ("semSlope", None), ("txtZ0", None), ("txtSlope", None), ("colorLedLexical", None), ("colorBoost", None)]


def get(p, key):
    a, b = key
    return p[a][b] if b else p[a]


def put(p, key, v):
    a, b = key
    if b:
        p[a][b] = v
    else:
        p[a] = v


def candidates_for(key, v):
    a, _ = key
    if a in ("semZ0", "txtZ0"):
        return [v + d for d in (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0)]
    if a in ("semSlope", "txtSlope"):
        return [max(0.25, v * m) for m in (0.5, 0.75, 1.33, 2.0)]
    base = v if v > 0 else 0.05
    return [0.0] + [base * m for m in (0.25, 0.5, 0.75, 1.33, 2.0, 4.0)]


def fit(qs, start: dict, passes: int = 4, seed: int = 0) -> dict:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--out")
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    qs = load(a.dump)
    print(f"{len(qs)} queries, {sum(len(q['cands']) for q in qs)} candidates")
    print(f"defaults: nDCG@10 {mean_ndcg(qs, DEFAULT):.4f}")
    by_cat = defaultdict(list)
    # cross-validation: fit on k-1 folds, test on the held-out one, stratified by category
    tests_fit, tests_def = [], []
    for rep in range(a.repeats):
        rng = random.Random(100 + rep)
        cats = defaultdict(list)
        for i, q in enumerate(qs):
            cats[q["cat"]].append(i)
        fold_of = {}
        for idx in cats.values():
            rng.shuffle(idx)
            for j, i in enumerate(idx):
                fold_of[i] = j % a.folds
        for f in range(a.folds):
            train = [q for i, q in enumerate(qs) if fold_of[i] != f]
            test = [q for i, q in enumerate(qs) if fold_of[i] == f]
            p = fit(train, DEFAULT, seed=rep * 10 + f)
            tests_fit.append(mean_ndcg(test, p))
            tests_def.append(mean_ndcg(test, DEFAULT))
            print(f"  repeat {rep} fold {f}: train {mean_ndcg(train, p):.4f}  held-out fitted {tests_fit[-1]:.4f}  held-out defaults {tests_def[-1]:.4f}")
    print(f"held-out mean: fitted {sum(tests_fit) / len(tests_fit):.4f}  defaults {sum(tests_def) / len(tests_def):.4f}")
    final = fit(qs, DEFAULT, passes=6)
    print(f"fitted on all: nDCG@10 {mean_ndcg(qs, final):.4f}")
    for q in qs:
        by_cat[q["cat"]].append((ndcg(q, DEFAULT), ndcg(q, final)))
    for c, v in by_cat.items():
        print(f"  {c:<13} defaults {sum(x for x, _ in v) / len(v):.3f}  fitted {sum(y for _, y in v) / len(v):.3f}")
    print(json.dumps(final, indent=1))
    if a.out:
        json.dump({**final, "version": "fitted-offline"}, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
