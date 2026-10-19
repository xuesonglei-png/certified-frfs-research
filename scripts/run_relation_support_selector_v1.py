"""Prospective nested relation--support co-design experiment.

This is an append-only experiment.  It uses three outer folds on the 15
development datasets, records every inner candidate, and evaluates the chosen
weighted relation on an outer test fold.  It is not a replacement for the
frozen primary tables until its independent replay passes.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from certified_frfs.datasets import load_dataset
from certified_frfs.relation_support_selector_v1 import fit_outer, weighted_knn
from certified_frfs.validated_pool import preprocess_fit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "relation-support-selector-v4"
DATASETS = ("banknote", "ecoli", "glass", "haberman", "heart_statlog",
            "ionosphere", "iris", "seeds", "sonar", "transfusion",
            "vehicle", "vertebral", "wdbc", "wine", "yeast")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_dataset(name):
    X, y = load_dataset(name)
    outer_k = min(3, int(np.unique(y, return_counts=True)[1].min()))
    outer = StratifiedKFold(outer_k, shuffle=True, random_state=20261011)
    rows = []
    for fold, (train, test) in enumerate(outer.split(X, y)):
        A, B = preprocess_fit(X[train], X[test])
        start = time.perf_counter()
        inner_k = min(3, int(np.unique(y[train], return_counts=True)[1].min()))
        if inner_k < 2:
            rows.append({"fold": fold, "status": "insufficient_inner_class_repetition",
                         "train_indices": train.tolist(), "test_indices": test.tolist(),
                         "inner_splits": inner_k})
            continue
        fit = fit_outer(X[train], y[train], seed=20261011 + fold,
                        inner_splits=inner_k)
        chosen = fit["selected"]
        fs = chosen["features"]
        w = np.asarray(chosen["weights"], float)
        proposed = weighted_knn(A, y[train], B, y[test], fs, w)
        matched = weighted_knn(A, y[train], B, y[test], fs,
                              np.ones(len(fs)) / np.sqrt(len(fs)))
        full = weighted_knn(A, y[train], B, y[test], list(range(A.shape[1])),
                            np.ones(A.shape[1]) / np.sqrt(A.shape[1]))
        # Shared perturbation draws evaluate robustness without entering choice.
        perturbed = []; matched_perturbed = []; full_perturbed = []
        for draw in range(8):
            rng = np.random.default_rng(20262000 + 10 * fold + draw)
            Q = np.clip(B + rng.uniform(-.01, .01, B.shape), 0, 1)
            perturbed.append(weighted_knn(A, y[train], Q, y[test], fs, w)["clean_BA"])
            matched_perturbed.append(weighted_knn(A, y[train], Q, y[test], fs,
                                                   np.ones(len(fs)) / np.sqrt(len(fs)))["clean_BA"])
            full_perturbed.append(weighted_knn(A, y[train], Q, y[test], list(range(A.shape[1])),
                                               np.ones(A.shape[1]) / np.sqrt(A.shape[1]))["clean_BA"])
        rows.append({"fold": fold, "status": "complete", "train_indices": train.tolist(), "test_indices": test.tolist(),
                     "fit": fit, "proposed": proposed, "matched_uniform": matched, "full": full,
                     "perturbed_BA": float(np.mean(perturbed)),
                     "matched_perturbed_BA": float(np.mean(matched_perturbed)),
                     "full_perturbed_BA": float(np.mean(full_perturbed)),
                     "seconds": time.perf_counter() - start})
        print(name, fold, {k: round(v["clean_BA"], 5) for k, v in
                           (("proposed", proposed), ("matched", matched), ("full", full))},
              "support", len(fs), "/", A.shape[1], flush=True)
    return {"dataset": name, "d": int(X.shape[1]), "rows": rows,
            "fingerprints": {"data": digest(ROOT / "data" / "processed" / f"{name}.npz"),
                              "runner": digest(Path(__file__))},
            "scope": "prospective_nested_relation_support_co_design_development"}


def main():
    threadpool_limits(1); OUT.mkdir(exist_ok=True)
    rows = []
    for name in DATASETS:
        path = OUT / f"{name}.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8"))); continue
        row = run_dataset(name)
        path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
    deltas = []
    robust_deltas = []
    matched_deltas = []
    removed = []
    for row in rows:
        for r in row["rows"]:
            if r.get("status") != "complete":
                continue
            deltas.append(100 * (r["proposed"]["clean_BA"] - r["full"]["clean_BA"]))
            robust_deltas.append(100 * (r["perturbed_BA"] - r["full_perturbed_BA"]))
            matched_deltas.append(100 * (r["proposed"]["clean_BA"] - r["matched_uniform"]["clean_BA"]))
            removed.append(1 - len(r["fit"]["selected"]["features"]) / row["d"])
    summary = {"datasets": len(rows), "outer_folds": len(deltas),
               "failed_folds": int(sum(r.get("status") != "complete" for row in rows for r in row["rows"])),
               "mean_clean_delta_vs_full_pp": float(np.mean(deltas)),
               "mean_perturbed_delta_vs_full_pp": float(np.mean(robust_deltas)),
               "mean_clean_delta_vs_matched_uniform_pp": float(np.mean(matched_deltas)),
               "wins_vs_full": int(sum(v > 1e-12 for v in deltas)),
               "losses_vs_full": int(sum(v < -1e-12 for v in deltas)),
               "wins_vs_matched_uniform": int(sum(v > 1e-12 for v in matched_deltas)),
               "mean_removed": float(np.mean(removed)),
               "goal_achieved": False,
               "independent_audit_required": True,
               "scope": "development_only_no_SOTA_or_population_claim"}
    (OUT / "complete.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
