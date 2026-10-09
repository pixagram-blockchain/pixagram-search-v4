#!/usr/bin/env python3
"""Search-quality evaluation against a deployed stack (or two, side by side).

    python3 scripts/eval.py https://pixagram-search-v3.p1x4.workers.dev
    python3 scripts/eval.py --compare https://pixagram-search-v2.p1x4.workers.dev https://pixagram-search-v3.p1x4.workers.dev
    python3 scripts/eval.py <url> --cat visual --show 5          # per-query detail for one category
    python3 scripts/eval.py <url> --ask                          # /ask questions (eval/ask.jsonl)

Judgments live in eval/queries.jsonl: one query per line with graded relevance (2 = what the
query is about, 1 = partially relevant) keyed by "author/permlink", so the same file scores any
stack whatever its post ids. Metrics per category and overall:

    nDCG@10   graded, the headline number
    P@10      share of the top 10 that is relevant (grade >= 1)
    R@10/R@50 share of the relevant items found in the top 10 / 50
    MRR       1 / rank of the first relevant item

Only the first 50 results are fetched (one page of the API).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = {"accept": "application/json", "user-agent": "pixagram-search-eval/3.0"}  # Cloudflare 403s urllib's default UA


def get_json(url: str, body: dict | None = None, tries: int = 3) -> dict:
    last: Exception | None = None
    for k in range(tries):
        try:
            data = json.dumps(body).encode() if body is not None else None
            headers = {**UA, **({"content-type": "application/json"} if body is not None else {})}
            req = urllib.request.Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 - report after retries
            last = e
            time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"{url}: {last}")


def items_of(d: dict) -> list[dict]:
    # v2/v3 return {"items": [...]}; the older production build nests {"artworks": {"items": [...]}}.
    if "items" in d:
        return d["items"]
    return (d.get("artworks") or {}).get("items", [])


def search(base: str, q: str, type_: str, extra: dict) -> tuple[list[str], dict]:
    params = {"q": q, "limit": 50, "nsfw": "include", **({"type": type_} if type_ else {}), **extra}
    d = get_json(f"{base.rstrip('/')}/search?" + urllib.parse.urlencode(params))
    refs = [f"{i['author']}/{i['permlink']}" for i in items_of(d)]
    return refs, d


def metrics(ranked: list[str], rel: dict[str, int]) -> dict[str, float]:
    gains = [rel.get(r, 0) for r in ranked]
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal)) or 1.0
    n_rel = sum(1 for v in rel.values() if v > 0) or 1
    first = next((i for i, g in enumerate(gains) if g > 0), None)
    return {
        "ndcg10": dcg / idcg,
        "p10": sum(1 for g in gains[:10] if g > 0) / 10,
        "r10": sum(1 for g in gains[:10] if g > 0) / n_rel,
        "r50": sum(1 for g in gains[:50] if g > 0) / n_rel,
        "mrr": 0.0 if first is None else 1.0 / (first + 1),
    }


KEYS = ("ndcg10", "p10", "r10", "r50", "mrr")


def mean(rows: list[dict]) -> dict[str, float]:
    return {k: sum(r[k] for r in rows) / len(rows) for k in KEYS} if rows else {k: 0.0 for k in KEYS}


def run(base: str, queries: list[dict], extra: dict, show: int, cat: str | None) -> dict:
    per_cat: dict[str, list[dict]] = {}
    detail = []
    for x in queries:
        if cat and x["cat"] != cat:
            continue
        try:
            refs, d = search(base, x["q"], x.get("type", "artwork"), extra)
            m = metrics(refs, x["rel"])
        except Exception as e:  # noqa: BLE001
            refs, d, m = [], {"error": str(e)}, {k: 0.0 for k in KEYS}
        per_cat.setdefault(x["cat"], []).append(m)
        detail.append({"q": x["q"], "cat": x["cat"], "m": m, "top": refs[:show], "mode": d.get("mode"), "rel": x["rel"], "notes": d.get("notes")})
    summary = {c: mean(rows) | {"n": len(rows)} for c, rows in per_cat.items()}
    summary["ALL"] = mean([m for rows in per_cat.values() for m in rows]) | {"n": sum(len(r) for r in per_cat.values())}
    return {"base": base, "summary": summary, "detail": detail}


def print_summary(results: list[dict]) -> None:
    cats = list(results[0]["summary"].keys())
    head = f"{'category':<13}{'n':>4}  " + "  ".join(f"{'nDCG@10':>8}{'P@10':>6}{'R@10':>6}{'R@50':>6}{'MRR':>6}" for _ in results)
    print(head)
    for c in cats:
        line = f"{c:<13}{results[0]['summary'][c]['n']:>4}  "
        line += "  ".join(
            f"{r['summary'][c]['ndcg10']:>8.3f}{r['summary'][c]['p10']:>6.2f}{r['summary'][c]['r10']:>6.2f}{r['summary'][c]['r50']:>6.2f}{r['summary'][c]['mrr']:>6.2f}"
            for r in results
        )
        print(line)
    for i, r in enumerate(results):
        print(f"  [{i + 1}] {r['base']}")


def run_ask(base: str, path: str) -> int:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    ok = 0
    for x in rows:
        try:
            d = get_json(f"{base.rstrip('/')}/ask", {"question": x["question"]})
        except Exception as e:  # noqa: BLE001
            print(f"ERR  {x['question']}: {e}")
            continue
        got = d.get("answer")
        exp = x["expect"]
        hit = False
        if "ref" in exp:
            ev = (d.get("evidence") or [{}])[0]
            hit = f"{ev.get('author')}/{ev.get('permlink')}" in ([exp["ref"]] if isinstance(exp["ref"], str) else exp["ref"])
        elif "author" in exp:
            hit = str(got).lstrip("@") == exp["author"]
        elif "count" in exp:
            hit = isinstance(got, (int, float)) and abs(got - exp["count"]) <= exp.get("tolerance", 0)
        elif "date" in exp:
            hit = got == exp["date"]
        ok += hit
        print(f"{'ok ' if hit else 'MISS'} {x['question']}\n     → {json.dumps(got, ensure_ascii=False)} (confidence {d.get('confidence')}; intent {(d.get('plan') or {}).get('intent')})")
    print(f"\n{ok}/{len(rows)} answered as expected")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", nargs="?", help="stack to evaluate")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="evaluate two stacks side by side")
    ap.add_argument("--queries", default=os.path.join(ROOT, "eval", "queries.jsonl"))
    ap.add_argument("--cat", help="only this category")
    ap.add_argument("--show", type=int, default=0, help="print the top N results of every query")
    ap.add_argument("--param", action="append", default=[], help="extra query-string parameter k=v (e.g. semantic=0)")
    ap.add_argument("--json", help="write full results to this file")
    ap.add_argument("--ask", action="store_true", help="run eval/ask.jsonl against /ask instead")
    args = ap.parse_args()
    if args.ask:
        if not args.base:
            ap.error("give the stack URL")
        return run_ask(args.base, os.path.join(ROOT, "eval", "ask.jsonl"))
    bases = args.compare or ([args.base] if args.base else [])
    if not bases:
        ap.error("give a stack URL or --compare A B")
    extra = dict(p.split("=", 1) for p in args.param)
    queries = [json.loads(l) for l in open(args.queries, encoding="utf-8") if l.strip()]
    results = [run(b, queries, extra, args.show, args.cat) for b in bases]
    if args.show:
        for i, x in enumerate(results[0]["detail"]):
            print(f"\n[{x['cat']}] {x['q']}")
            for r in results:
                y = r["detail"][i]
                marks = " ".join(f"{t.split('/')[1][:24]}{'**' if y['rel'].get(t) == 2 else '*' if y['rel'].get(t) else ''}" for t in y["top"])
                print(f"  nDCG {y['m']['ndcg10']:.2f}  {marks}")
    print_summary(results)
    if args.json:
        json.dump(results, open(args.json, "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
