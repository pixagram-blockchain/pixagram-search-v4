#!/usr/bin/env python3
"""
Put the stack's Hugging Face Space on the hardware and sleep time it should run on.
Used by scripts/deploy.sh and scripts/deploy-space.sh.

    python3 scripts/space_hardware.py <space id> [hardware] [sleep]

hardware: a Space hardware name (default cpu-upgrade: 8 vCPU, 32 GB, $0.03 an hour), or "keep"
          to leave it as it is.
sleep:    seconds without a request before the Space sleeps; -1 (default) never. Paid hardware
          only: free CPU basic always sleeps after 48 hours.

Why (v4.8.2): on free CPU basic the v4 Space stopped answering from 29 October to 3 November 2026
(found asleep), and the artworks posted meanwhile got no vectors. Never sleeping takes paid hardware, and on CPU upgrade
the Space computes eight embeddings at once instead of two (hf/siglip.py sizes itself to its CPUs;
the Worker reads that from /health).

Changing either restarts the Space; nothing is requested when both are already right. Needs an HF
login with write access to the Space (hf auth login) and a payment method on the account for
paid hardware.
"""

from __future__ import annotations

import sys

from huggingface_hub import HfApi


def _name(x: object) -> str:
    return str(getattr(x, "value", x) or "")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    sid = sys.argv[1]
    hw = (sys.argv[2] if len(sys.argv) > 2 else "") or "cpu-upgrade"
    sleep = int((sys.argv[3] if len(sys.argv) > 3 else "") or "-1")
    if hw == "keep":
        print("    hardware left as it is (SPACE_HARDWARE=keep)", file=sys.stderr)
        return 0
    api = HfApi()
    rt = api.get_space_runtime(sid)
    current = _name(rt.requested_hardware) or _name(rt.hardware)
    free = hw == "cpu-basic"
    said = "never" if sleep < 0 else f"{sleep} s"
    # on paid hardware an unset sleep time means never
    sleep_ok = free or (rt.sleep_time in (None, -1) if sleep < 0 else rt.sleep_time == sleep)
    if current != hw:
        api.request_space_hardware(sid, hw, sleep_time=None if free else sleep)
        print(f"    hardware {current or '?'} → {hw}{'' if free else f', sleep {said}'} (the Space restarts)", file=sys.stderr)
    elif not sleep_ok:
        api.set_space_sleep_time(sid, sleep)
        print(f"    hardware {hw}; sleep time {rt.sleep_time} s → {said} (the Space restarts)", file=sys.stderr)
    else:
        print(f"    hardware {hw}{'' if free else f', sleep {said}'}: already so", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
