"""Shared helpers for Paper-5 figure/table generation (read-only, JSON-sourced).

Governing rule (supervisor): EVERY number in EVERY figure/table must come from a
committed ``results/*.json`` cell. No hardcoded numbers, nothing typed from memory.
The ``Audit`` object enforces this: the only sanctioned way to pull a value is
``aud.val(json_path, "dotted.key")`` (or ``aud.arr(...)`` for a list), which (a) loads
the JSON, (b) navigates to the exact cell, (c) logs ``path :: key -> value`` to an audit
trail, and (d) returns it. Each script writes its audit trail next to its output so a
reviewer can re-trace every plotted/tabulated number to a JSON cell.

No training, no data generation, no GPU — pure post-processing of banked results.
"""
from __future__ import annotations

import functools
import glob
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(_ROOT, "results")
FIG_DIR = os.path.join(_ROOT, "paper", "figures")
TAB_DIR = os.path.join(_ROOT, "paper", "tables")
AUD_DIR = os.path.join(_ROOT, "paper", "audit")
SEEDS = (11, 22, 33, 44, 55)


@functools.lru_cache(maxsize=None)
def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _navigate(d, dotted: str):
    cur = d
    for seg in dotted.split("."):
        if isinstance(cur, (list, tuple)):
            cur = cur[int(seg)]
        else:
            cur = cur[seg]
    return cur


class Audit:
    """Accumulates (source_json, cell, value) and writes a re-traceable trail."""

    def __init__(self, name: str):
        self.name = name
        self.rows: list[tuple[str, str, object]] = []

    def val(self, json_path: str, dotted: str):
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"[{self.name}] MISSING JSON (cannot source from memory): {json_path}")
        v = _navigate(_load(json_path), dotted)
        self.rows.append((os.path.relpath(json_path, _ROOT), dotted, v))
        return v

    def arr(self, json_path: str, dotted: str) -> list:
        v = self.val(json_path, dotted)
        if not isinstance(v, list):
            raise TypeError(f"[{self.name}] expected list at {dotted}, got {type(v)}")
        # overwrite the scalar-style log row with a length summary
        self.rows[-1] = (os.path.relpath(json_path, _ROOT), dotted, f"<list n={len(v)}>")
        return v

    def note(self, msg: str):
        self.rows.append(("(derived)", msg, ""))

    def write(self):
        os.makedirs(AUD_DIR, exist_ok=True)
        out = os.path.join(AUD_DIR, f"{self.name}.audit.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"# Consistency audit — {self.name}\n")
            f.write("# Every figure/table number traces to a results/*.json cell below.\n\n")
            f.write(f"{'source JSON':<62} {'cell':<34} value\n")
            f.write("-" * 110 + "\n")
            for src, cell, v in self.rows:
                f.write(f"{src:<62} {cell:<34} {v}\n")
        print(f"  audit -> {os.path.relpath(out, _ROOT)} ({len(self.rows)} cells)")
        return out


def metrics_path(family: str, dtag: str, seed: int) -> str:
    """results/indomain/<family>_<dtag>_seed<seed>_metrics.json (headline eval)."""
    return os.path.join(RESULTS, "indomain", f"{family}_{dtag}_seed{seed}_metrics.json")


def have(path: str) -> bool:
    return os.path.exists(path)


IEEE_STYLE = os.path.join(os.path.dirname(__file__), "figures", "ieee_style.mplstyle")


def setup_mpl():
    """IEEE TVT / Vehicular Communications figure style (single chokepoint for all figures).

    Loads scripts/figures/ieee_style.mplstyle. STYLING ONLY — does not touch any data path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("default")          # clean base so the style fully determines appearance
    plt.style.use(IEEE_STYLE)
    if "Times New Roman" not in {f.name for f in
                                 matplotlib.font_manager.fontManager.ttflist}:
        print("  [style] Times New Roman not found; falling back to STIX/DejaVu serif.")
    return plt


def save_fig(fig, stem: str):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{stem}.{ext}"))
    print(f"  figure -> paper/figures/{stem}.pdf (+png)")


def write_table(stem: str, tex: str, md: str):
    os.makedirs(TAB_DIR, exist_ok=True)
    with open(os.path.join(TAB_DIR, f"{stem}.tex"), "w", encoding="utf-8") as f:
        f.write(tex)
    with open(os.path.join(TAB_DIR, f"{stem}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  table -> paper/tables/{stem}.tex (+md)")
