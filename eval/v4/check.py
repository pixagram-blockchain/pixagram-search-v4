#!/usr/bin/env python3
"""Check the v4 question set (CI): every line well-formed, ids unique, the categories present, at
least 1,000 questions (spec §62).

    python3 -I eval/v4/check.py src/evaluation/datasets/questions.jsonl
"""
import collections
import json
import sys

CATEGORIES = ["factual", "semantic", "visual", "temporal", "comparative", "multi_hop", "ambiguous", "multilingual", "adversarial"]
TYPES = {"author", "date", "count", "boolean", "post", "duration", "value", "status", "retrieval"}
LANGS = {"en", "fr", "de", "es", "it"}


def main(path: str) -> int:
    rows, errors, ids = [], [], set()
    for i, line in enumerate(open(path, encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError as e:
            errors.append(f"line {i}: not JSON ({e})")
            continue
        for k in ("id", "question", "query_type", "lang", "answer_type", "expected_answer", "acceptable_answers", "required_evidence"):
            if k not in r:
                errors.append(f"line {i}: no {k}")
        if r.get("id") in ids:
            errors.append(f"line {i}: duplicate id {r.get('id')}")
        ids.add(r.get("id"))
        if r.get("query_type") not in CATEGORIES:
            errors.append(f"line {i}: unknown query_type {r.get('query_type')}")
        if r.get("answer_type") not in TYPES:
            errors.append(f"line {i}: unknown answer_type {r.get('answer_type')}")
        if r.get("lang") not in LANGS:
            errors.append(f"line {i}: unknown lang {r.get('lang')}")
        if r.get("answer_type") == "status" and not r.get("expected_status"):
            errors.append(f"line {i}: a status question needs expected_status")
        if r.get("answer_type") == "retrieval" and not r.get("rel"):
            errors.append(f"line {i}: a retrieval question needs rel")
        if r.get("query_type") == "visual" and not r.get("image"):
            errors.append(f"line {i}: a visual question needs an image")
        rows.append(r)
    by = collections.Counter(r.get("query_type") for r in rows)
    print(f"{len(rows)} questions: " + ", ".join(f"{k} {by.get(k, 0)}" for k in CATEGORIES))
    if len(rows) < 1000:
        errors.append(f"only {len(rows)} questions (at least 1,000)")
    for k in CATEGORIES:
        if not by.get(k):
            errors.append(f"no {k} question")
    for e in errors[:50]:
        print("ERROR", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "src/evaluation/datasets/questions.jsonl"))
