"""Fresh shared-reference/query stress draws on frozen audited atomic models.

No fitting, support search, or selection takes place in this script.  Predictions
are independently rebuilt with NumPy city-block distances, stable tie handling,
and majority votes.  This is a development stress test, not confirmation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "thinned-atomic-classifier-v2"
OUT = ROOT / "results" / "atomic-frozen-stress-v1"
METHODS = ("proposed", "full_refined", "fgs_refined", "sfrip_refined",
           "full_source", "fgs_native", "sfrip_native", "fgs_exact", "sfrip_exact")
SEEDS = tuple(range(20261200, 20261208))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preprocess(reference, query):
    med = np.array([np.median(v[~np.isnan(v)]) if np.any(~np.isnan(v)) else 0 for v in reference.T])
    A = np.where(np.isnan(reference), med, reference)
    B = np.where(np.isnan(query), med, query)
    lo, hi = A.min(axis=0), A.max(axis=0)
    scale = hi-lo
    scale[scale == 0] = 1
    return np.clip((A-lo)/scale, 0, 1), np.clip((B-lo)/scale, 0, 1)


def predict(A, labels, B, features, weights):
    classes = np.unique(labels)
    if not features:
        c = classes[np.argmax([np.sum(labels == k) for k in classes])]
        return np.repeat(c, len(B))
    fs, w = list(features), np.asarray(weights)
    # Deliberately rebuild without the production scorer or cdist routine.
    distances = np.abs(B[:, None, fs]*w-A[None, :, fs]*w).sum(axis=2)
    nn = np.argsort(distances, kind="stable", axis=1)[:, :5]
    votes = np.stack([(labels[nn] == c).sum(axis=1) for c in classes], axis=1)
    return classes[votes.argmax(axis=1)]


def ba(pred, truth):
    return float(np.mean([np.mean(pred[truth == c] == c) for c in np.unique(truth)]))


def main():
    OUT.mkdir(exist_ok=True)
    source_audit = json.loads((SOURCE/"complete-audit.json").read_text())
    assert source_audit["passed"]
    rows = []
    for p in sorted(SOURCE.glob("*.json")):
        name = p.stem
        if name.startswith("complete") or name.endswith("-audit"):
            continue
        saved = json.loads(p.read_text())
        with np.load(ROOT/"data"/"processed"/(name+".npz")) as data:
            X, y = data["X"], data["y"]
        ri, vi = np.array(saved["reference_indices"]), np.array(saved["validation_indices"])
        assert not set(ri) & set(vi)
        A, B = preprocess(X[ri], X[vi])
        models = saved["outer"]["results"]
        item = {"dataset": name, "source_sha256": digest(p), "metrics": {}}
        for method in METHODS:
            model = models[method]
            metric = model["metric"]
            pred = predict(A, y[ri], B, metric["features"], metric["weights"])
            assert pred.tolist() == model["evaluation"]["predictions"], (name, method, "clean replay mismatch")
            vals = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                # Common perturbation tensors are regenerated identically for
                # every method; their shape covers all original coordinates.
                PA = np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1)
                PB = np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1)
                vals.append(ba(predict(PA, y[ri], PB, metric["features"], metric["weights"]), y[vi]))
            item["metrics"][method] = {"clean_BA": ba(pred, y[vi]), "shared_perturbed_BA": float(np.mean(vals)), "worst_draw_BA": float(np.min(vals)), "draws": vals, "features": len(metric["features"])}
        rows.append(item)
        (OUT/(name+".json")).write_text(json.dumps(item, indent=2), encoding="utf8")
        print(name, "clean prediction replay and 72 new stress evaluations passed", flush=True)
    comparisons = {}
    for method in METHODS[1:]:
        comparisons[method] = {}
        for key in ("clean_BA", "shared_perturbed_BA", "worst_draw_BA"):
            diff = np.array([r["metrics"]["proposed"][key]-r["metrics"][method][key] for r in rows])
            nz = diff[np.abs(diff) > 1e-12]
            comparisons[method][key] = {"mean_delta_pp": float(diff.mean()*100), "wins": int(np.sum(diff > 1e-12)), "losses": int(np.sum(diff < -1e-12)), "ties": int(np.sum(np.abs(diff) <= 1e-12)), "wilcoxon_two_sided_p": float(wilcoxon(nz).pvalue) if len(nz) else 1.0}
    result = {"rows": rows, "comparisons": comparisons, "datasets": len(rows), "source_audit_sha256": digest(SOURCE/"complete-audit.json"), "runner_sha256": digest(Path(__file__)), "stress_seeds": SEEDS, "clean_replay_passed": True, "new_stress_evaluations": len(rows)*len(METHODS)*len(SEEDS), "goal_achieved": False, "scope": "append_only_frozen_audited_models_new_shared_reference_query_draws_exposed_development_not_confirmation"}
    (OUT/"complete.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
