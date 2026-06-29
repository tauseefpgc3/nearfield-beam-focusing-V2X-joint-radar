"""Patient DeepMIMO scenario downloader (handles 429 rate-limiting).

Helper for data setup only (not a deliverable). Waits out the server cooldown,
downloads o1_60 then o1_140 with spacing, and retries on transient failures.
"""
import time

import deepmimo as dm

SCENARIOS = ["o1_60", "o1_140"]
MAX_RETRIES = 8
COOLDOWN_S = 180  # initial wait to let any active 429 window expire


def fetch(name: str) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            p = dm.download(name)
        except Exception as e:  # noqa: BLE001 - report, then retry
            print(f"[{name}] attempt {attempt}: exception {e!r}", flush=True)
            p = None
        if p:
            print(f"[{name}] DONE -> {p}", flush=True)
            return True
        wait = min(60 * attempt, 600)
        print(f"[{name}] attempt {attempt} failed (None); sleeping {wait}s", flush=True)
        time.sleep(wait)
    print(f"[{name}] GAVE UP after {MAX_RETRIES} attempts", flush=True)
    return False


if __name__ == "__main__":
    print(f"Initial cooldown {COOLDOWN_S}s to clear rate-limit window...", flush=True)
    time.sleep(COOLDOWN_S)
    for i, scen in enumerate(SCENARIOS):
        ok = fetch(scen)
        if i + 1 < len(SCENARIOS):
            time.sleep(120)  # space requests to avoid re-triggering 429
    print("DOWNLOADER_DONE", flush=True)
