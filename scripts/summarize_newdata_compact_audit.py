"""Append-only paired audit for the frozen eight-task new-data assessment.

This consumes the existing ``complete.json`` and writes a compact statistical
summary.  It intentionally reports descriptive paired tests only: eight tasks
are too few for a universal or confirmatory superiority claim.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "newdata-compact" / "complete.json"
AUDIT = ROOT / "results" / "newdata-compact" / "complete-audit.json"
OUT = ROOT / "results" / "newdata-compact" / "paired-audit-summary.json"


def paired(values: list[float]) -> dict:
    x = np.asarray(values, dtype=float)
    tol = 1e-12
    wins = int(np.sum(x > tol))
    ties = int(np.sum(np.abs(x) <= tol))
    losses = int(np.sum(x < -tol))
    nonzero = x[np.abs(x) > tol]
    p = float(wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0
    # The interval is descriptive and deterministic; it is not a confirmation
    # interval because the eight tasks are a fixed, exposed benchmark panel.
    rng = np.random.default_rng(20261012)
    draws = rng.choice(x, size=(20000, len(x)), replace=True).mean(axis=1)
    return {
        "mean_delta": float(x.mean()),
        "mean_delta_pp": float(100 * x.mean()),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "wilcoxon_two_sided_p": p,
        "bootstrap95_mean_delta_pp": [float(v) for v in np.percentile(100 * draws, [2.5, 97.5])],
    }


def main() -> None:
    source_bytes = SOURCE.read_bytes()
    source = json.loads(source_bytes.decode("utf-8-sig"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8-sig"))
    rows = source["rows"]
    methods = ("full", "fgs2023", "sfrip2025")
    comparisons = {}
    for method in methods:
        comparisons[method] = {
            "clean": paired([r["metrics"]["proposed"]["clean"] - r["metrics"][method]["clean"] for r in rows]),
            "noise01": paired([r["metrics"]["proposed"]["noise01"] - r["metrics"][method]["noise01"] for r in rows]),
            "utility": paired([r["utility"]["proposed"] - r["utility"][method] for r in rows]),
        }
    speed = {}
    for method in ("fgs2023", "sfrip2025"):
        ratios = [r["seconds"][method] / r["seconds"]["proposed"] for r in rows]
        speed[method] = {
            "mean_baseline_over_proposed": float(np.mean(ratios)),
            "median_baseline_over_proposed": float(np.median(ratios)),
        }
    result = {
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_audit": audit,
        "datasets": [r["dataset"] for r in rows],
        "n_datasets": len(rows),
        "comparisons": comparisons,
        "mean_removed_proposed": float(np.mean([r["removed"]["proposed"] for r in rows])),
        "mean_removed_proposed_pp": float(100 * np.mean([r["removed"]["proposed"] for r in rows])),
        "speed_ratios": speed,
        "interpretation": "Descriptive held-out assessment only; n=8 and the source protocol does not provide a fully nested optimizer refit or confirmatory inference.",
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
