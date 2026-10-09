#!/usr/bin/env python3
"""Model-only retrieval quality, without the ranker: every judged query (eval/queries.jsonl) ranks
the posts of its type (all artworks so far) by cosine(query vector, image vector), and by
cosine(query vector, text vector). nDCG@10 and R@10 per category, for one or more offline-harness
states.

    # after a run of eval/offline/run.test.ts with STATE=/tmp/base.sqlite, with the same model's
    # embed server still running:
    python3 eval/offline/vectors_only.py /tmp/base.sqlite [/tmp/other.sqlite ...] [--embed-url URL]

The query vectors come from the embed server (POST /embed) and are cached next to each state
(<state>.queries.json); a state whose cache exists does not need its server any more. Delete the
cache when the state is rebuilt with another model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import urllib.request
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
QUERIES = [json.loads(line) for line in open(os.path.join(HERE, "..", "queries.jsonl")) if line.strip()]
CATS = ["visual", "multilingual", "tag", "ambiguous", "color", "title", "typo", "author", "all"]


def query_vectors(state: str, url: str) -> dict:
    cache = f"{state}.queries.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    texts = sorted({q["q"] for q in QUERIES})
    vecs: dict = {}
    for i in range(0, len(texts), 32):
        batch = texts[i : i + 32]
        req = urllib.request.Request(url, data=json.dumps({"inputs": {"texts": batch}}).encode(), headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            vecs.update(zip(batch, json.load(r)["embeddings"]))
    json.dump(vecs, open(cache, "w"))
    return vecs


def ndcg(ranked: list, rel: dict, k: int = 10) -> float:
    dcg = sum((2 ** rel.get(r, 0) - 1) / math.log2(i + 2) for i, r in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall(ranked: list, rel: dict, k: int) -> float:
    good = {r for r, g in rel.items() if g > 0}
    return len(good & set(ranked[:k])) / len(good) if good else 0.0


def scores(state: str, url: str) -> dict:
    db = sqlite3.connect(state)
    ref = {str(i): (f"{a}/{p}", t) for i, a, p, t in db.execute("SELECT id, author, permlink, type FROM posts")}
    saved = json.load(open(f"{state}.vectors.json"))
    qv = query_vectors(state, url)
    out = {}
    for kind in ("image", "text"):
        ids = [k for k in saved[kind] if k in ref]
        mat = [saved[kind][k]["values"] for k in ids]
        dims = {len(m) for m in mat} | {len(v) for v in qv.values()}
        if len(dims) != 1:
            raise SystemExit(f"{state}: {kind} vectors and query vectors differ in dimension {sorted(dims)}: a cache from another model?")
        per: dict = defaultdict(list)
        for x in QUERIES:
            q = qv[x["q"]]
            cands = [(i, m) for i, m in zip(ids, mat) if ref[i][1] == x.get("type", "artwork")]
            ranked = [ref[i][0] for _, i in sorted(((sum(a * b for a, b in zip(q, m)), i) for i, m in cands), reverse=True)]
            m = (ndcg(ranked, x["rel"]), recall(ranked, x["rel"], 10))
            per[x["cat"]].append(m)
            per["all"].append(m)
        out[kind] = {c: (sum(a for a, _ in v) / len(v), sum(b for _, b in v) / len(v), len(v)) for c, v in per.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("states", nargs="+")
    ap.add_argument("--embed-url", default="http://127.0.0.1:7861/embed")
    a = ap.parse_args()
    res = {s: scores(s, a.embed_url) for s in a.states}
    names = [os.path.basename(s).removesuffix(".sqlite") for s in a.states]
    for kind in ("image", "text"):
        print(f"\n{kind} vectors only: nDCG@10 (R@10)")
        print("category".ljust(13) + "n".rjust(4) + "".join(n.rjust(16) for n in names))
        for c in CATS:
            first = res[a.states[0]][kind].get(c)
            if not first:
                continue
            cells = "".join(f"{res[s][kind][c][0]:.3f} ({res[s][kind][c][1]:.2f})".rjust(16) for s in a.states)
            print(c.ljust(13) + str(first[2]).rjust(4) + cells)


if __name__ == "__main__":
    main()
