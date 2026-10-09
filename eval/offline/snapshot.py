#!/usr/bin/env python3
"""Snapshot the Pixa chain for the offline evaluation (eval/offline/run.test.ts).

    python3 eval/offline/snapshot.py --out eval/snapshot          # ~10 MB, a few minutes

Writes
  chain_posts.json     every top-level post, as condenser_api.get_content returns it
  chain_history.json   {account: [[seq, entry], ...]} the top-level comment operations of every
                       account that posted (account_history_api, operation_filter_low = 2), i.e.
                       every create and edit with its full body, including posts deleted since

Accounts come from condenser_api.lookup_accounts and posts from Hivemind's
bridge.get_account_posts, the same walk as the Worker's backfill. The judged queries
(eval/queries.jsonl, eval/ask.jsonl) are keyed by author/permlink and were written against the
chain as of October 2026; posts added later only add candidates.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request

UA = {"content-type": "application/json", "user-agent": "pixagram-search-snapshot/3.0"}


class Rpc:
    def __init__(self, urls: list[str]):
        self.urls = urls
        self.n = 0

    def call(self, method: str, params, tries: int = 4):
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
        last: Exception | None = None
        for k in range(tries):
            url = self.urls[k % len(self.urls)]
            try:
                req = urllib.request.Request(url, data=body, headers=UA, method="POST")
                with urllib.request.urlopen(req, timeout=60) as r:
                    d = json.load(r)
                self.n += 1
                if "error" in d:
                    raise RpcError(json.dumps(d["error"]))
                return d["result"]
            except RpcError:
                raise
            except Exception as e:  # noqa: BLE001 - network: retry on the next node
                last = e
                time.sleep(1 + k)
        raise RuntimeError(f"{method}: {last}")


class RpcError(Exception):
    pass


def accounts(rpc: Rpc) -> list[str]:
    out, start = [], ""
    while True:
        page = rpc.call("condenser_api.lookup_accounts", [start, 1000])
        new = [a for a in page if a != start]
        out += new
        if len(page) < 1000 or not new:
            return out
        start = page[-1]


def account_posts(rpc: Rpc, account: str) -> list[tuple[str, str]]:
    refs, start = [], None
    while True:
        params = {"sort": "posts", "account": account, "limit": 20, "observer": ""}
        if start:
            params.update(start_author=start[0], start_permlink=start[1])
        try:
            page = rpc.call("bridge.get_account_posts", params)
        except RpcError as e:
            if "does not exist" in str(e):  # unknown to Hivemind (system accounts)
                return refs
            raise
        page = [p for p in page if (p["author"], p["permlink"]) != start]
        refs += [(p["author"], p["permlink"]) for p in page if not p.get("parent_author")]
        if len(page) < 19:
            return refs
        start = (page[-1]["author"], page[-1]["permlink"])


def comment_history(rpc: Rpc, account: str, max_calls: int = 400) -> list:
    """Top-level comment ops by the account, oldest first ([seq, entry] like the RPC)."""
    out, start, calls = [], -1, 0
    while calls < max_calls:
        calls += 1
        limit = 100 if start < 0 else min(100, start + 1)  # the node asserts start >= limit - 1
        try:
            r = rpc.call("account_history_api.get_account_history", {"account": account, "start": start, "limit": limit, "operation_filter_low": 2})
        except RpcError as e:
            m = re.search(r"set start=(\d+)", str(e))  # the 2000-operation scan cap
            if not m:
                raise
            start = int(m.group(1))
            continue
        h = r.get("history") or []
        for seq, e in h:
            op = e["op"]
            typ, val = (op["type"], op["value"]) if isinstance(op, dict) else (op[0], op[1])
            if typ in ("comment_operation", "comment") and val.get("parent_author") == "" and val.get("author") == account:
                out.append([seq, e])
        if not h:
            break
        low = min(seq for seq, _ in h)
        if low <= 0:
            break
        start = low - 1
    out.sort(key=lambda x: x[0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="eval/snapshot")
    ap.add_argument("--rpc", default="https://api.pixagram.com,https://merlion.surf,https://blockforge.lol")
    ap.add_argument("--accounts", help="comma list instead of every account")
    a = ap.parse_args()
    rpc = Rpc([u.strip() for u in a.rpc.split(",") if u.strip()])
    os.makedirs(a.out, exist_ok=True)

    names = a.accounts.split(",") if a.accounts else accounts(rpc)
    print(f"{len(names)} accounts", file=sys.stderr)
    posts, authors = [], []
    for name in names:
        refs = account_posts(rpc, name)
        if not refs:
            continue
        authors.append(name)
        for author, permlink in refs:
            p = rpc.call("condenser_api.get_content", [author, permlink])
            if p and p.get("author"):
                posts.append(p)
        print(f"  {name}: {len(refs)} posts", file=sys.stderr)
    json.dump(posts, open(os.path.join(a.out, "chain_posts.json"), "w"))

    history = {}
    for name in authors:
        history[name] = comment_history(rpc, name)
        print(f"  {name}: {len(history[name])} comment ops", file=sys.stderr)
    json.dump(history, open(os.path.join(a.out, "chain_history.json"), "w"))
    print(f"{len(posts)} posts by {len(authors)} authors, {sum(len(v) for v in history.values())} history ops, {rpc.n} RPC calls → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
