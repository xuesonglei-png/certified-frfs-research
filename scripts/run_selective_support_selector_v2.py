"""Nested pilot for certified selective-support selection."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from certified_frfs.datasets import load_dataset
from certified_frfs.selective_support_selector_v2 import fit_outer, _query_scores
from certified_frfs.validated_pool import preprocess_fit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "selective-support-selector-v2"
DATASETS = ("banknote", "ecoli", "glass", "heart_statlog", "ionosphere")


def run_dataset(name):
    X, y = load_dataset(name)
    outer_k = min(3, int(np.unique(y, return_counts=True)[1].min()))
    outer = StratifiedKFold(outer_k, shuffle=True, random_state=20261011)
    rows = []
    for fold, (train, test) in enumerate(outer.split(X, y)):
        inner_k = min(3, int(np.unique(y[train], return_counts=True)[1].min()))
        if inner_k < 2:
            rows.append({"fold": fold, "status": "insufficient_inner",
                         "train_indices": train.tolist(), "test_indices": test.tolist()})
            continue
        fit = fit_outer(X[train], y[train], seed=20261011 + fold,
                        inner_splits=inner_k)
        A, B = preprocess_fit(X[train], X[test])
        selected = fit["selected"]
        proposed = _query_scores(A, y[train], B, y[test], selected)
        d = A.shape[1]
        full = {"features": list(range(d)), "weights": (np.ones(d) / np.sqrt(d)).tolist()}
        full_score = _query_scores(A, y[train], B, y[test], full)
        fs = selected["features"]
        matched = {"features": fs, "weights": (np.ones(len(fs)) / np.sqrt(len(fs))).tolist()}
        matched_score = _query_scores(A, y[train], B, y[test], matched)
        rows.append({"fold": fold, "status": "complete",
                     "train_indices": train.tolist(), "test_indices": test.tolist(),
                     "fit": fit, "proposed": proposed, "matched_uniform": matched_score,
                     "full": full_score})
        print(name, fold, "clean/certified BA",
              *(round(100 * r[k], 3) for r in (proposed, matched_score, full_score)
                for k in ("clean_BA", "query_certified_BA")),
              "coverage", round(proposed["query_coverage"], 3),
              "support", len(fs), "/", d, flush=True)
    return {"dataset": name, "d": int(X.shape[1]), "rows": rows,
            "scope": "nested_development_pilot_certified_selective_support"}


def main():
    threadpool_limits(1); OUT.mkdir(exist_ok=True)
    artifacts = []
    for name in DATASETS:
        p = OUT / f"{name}.json"
        if p.exists():
            artifacts.append(json.loads(p.read_text(encoding="utf-8")))
        else:
            item = run_dataset(name); p.write_text(json.dumps(item, indent=2), encoding="utf-8")
            artifacts.append(item)
    rows = [r for a in artifacts for r in a["rows"] if r.get("status") == "complete"]
    def mean(method, key): return float(np.mean([r[method][key] for r in rows]))
    def delta(key, method="full"): return 100 * (mean("proposed", key) - mean(method, key))
    summary = {
        "datasets": len(artifacts), "outer_folds": len(rows),
        "mean_clean_BA": {m: mean(m, "clean_BA") for m in ("proposed", "matched_uniform", "full")},
        "mean_query_certified_BA": {m: mean(m, "query_certified_BA") for m in ("proposed", "matched_uniform", "full")},
        "mean_query_coverage": {m: mean(m, "query_coverage") for m in ("proposed", "matched_uniform", "full")},
        "delta_clean_vs_full_pp": delta("clean_BA"),
        "delta_certified_vs_full_pp": delta("query_certified_BA"),
        "delta_coverage_vs_full_pp": delta("query_coverage"),
        "mean_removed": float(np.mean([1-len(r["fit"]["selected"]["features"])/a["d"]
                                        for a in artifacts for r in a["rows"] if r.get("status") == "complete"])),
        "scope": "development_pilot_no_SOTA_or_confirmatory_claim",
    }
    (OUT / "complete.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
