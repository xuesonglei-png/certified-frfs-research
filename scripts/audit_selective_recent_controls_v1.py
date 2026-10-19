"""Independent split, nested choice, weighted votes and query-box replay.

The documented baseline path engines are rerun, but author-code fidelity is
not asserted. Query certificates use coordinate endpoint arithmetic independent
of the production triplet-bound routine.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from certified_frfs.datasets import load_dataset
from audit_multiscale_certified_metric_knn_pool import independent_evaluate, query_certificate
from audit_relation_support_selector_v1 import independent_evaluate as independent_raw_votes
from run_selective_recent_controls_v1 import OUT, ROOT, paths


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scaling(reference, query):
    med = np.array([np.median(v[np.isfinite(v)]) if np.any(np.isfinite(v)) else 0. for v in reference.T])
    A = np.where(np.isnan(reference), med, reference)
    B = np.where(np.isnan(query), med, query)
    lo, hi = A.min(axis=0), A.max(axis=0)
    width = hi-lo
    width[width == 0] = 1.
    # Explicit sklearn-equivalent multiplication order preserves float ties.
    scale = 1/width
    offset = -lo*scale
    return np.clip(A*scale+offset, 0, 1), np.clip(B*scale+offset, 0, 1)


def selected_index(inners, method):
    available = []
    for i in range(len(inners[0]["options"][method])):
        entries = [b["options"][method][i] for b in inners]
        if any(e["score"] is None for e in entries):
            continue
        available.append((i, np.mean([e["score"]["clean_BA"] for e in entries]),
                          np.mean([e["score"]["query_certified_BA"] for e in entries]),
                          np.mean([e["score"]["query_coverage"] for e in entries]),
                          np.mean([len(e["metric"]["features"]) for e in entries])))
    if not available:
        return None
    best = max(v[1] for v in available)
    available = [v for v in available if v[1] >= best-.005-1e-12]
    return max(available, key=lambda v: (v[2], v[3], v[1], -v[4], -v[0]))[0]


def score_replay(A, y, B, truth, model, counts, cache):
    metric, stored = model["metric"], model["score"]
    fs, w = metric["features"], metric["weights"]
    assert sorted(fs) == fs and len(fs) == len(set(fs))
    if w is None:
        assert metric["status"] == "infeasible_original_2_over_d_unit_mass_cap"
        assert 2*len(fs) < A.shape[1] and stored is None
        counts["explicit_infeasible_cap_options"] += 1
        return
    assert len(w) == len(fs) and all(np.isfinite(w)) and all(v >= 0 for v in w)
    if metric["variant"].endswith("capped_transfer") and fs:
        mass = np.sqrt(A.shape[1])
        np.testing.assert_allclose(sum(w), mass, rtol=0, atol=1e-9)
        assert max(w) <= 2*mass/A.shape[1]+1e-9
        counts["capped_mass_checks"] += 1
    key = (tuple(fs), tuple(w))
    if key not in cache:
        if fs:
            raw = independent_raw_votes(A, y, B, truth, fs, w)
            replay = {"predictions": raw["predictions"], "probabilities": raw["probabilities"],
                      "metrics": {k: raw[k] for k in ("clean_BA", "balanced_logloss")}}
        else:
            replay = independent_evaluate(A, y, B, truth, fs, w)
        pred = np.array(replay["predictions"])
        certificate = query_certificate(A, y, B, pred, fs, w)
        correct = (pred == truth) & np.array(certificate["certified"])
        cba = float(np.mean([np.mean(correct[truth == c]) for c in np.unique(truth)]))
        cache[key] = (replay, cba, certificate["coverage"] if "coverage" in certificate else 0.)
        counts["independent_metric_prediction_replays"] += 1
    replay, cba, coverage = cache[key]
    assert stored["predictions"] == replay["predictions"]
    np.testing.assert_allclose(stored["probabilities"], replay["probabilities"], atol=1e-12, rtol=1e-12)
    for k in ("clean_BA", "balanced_logloss"):
        np.testing.assert_allclose(stored[k], replay["metrics"][k], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose([stored["query_certified_BA"], stored["query_coverage"]], [cba, coverage], atol=1e-12)
    counts["option_prediction_checks"] += 1


def block_replay(A, y, B, truth, saved, counts):
    fresh = json.loads(json.dumps(paths(A, y)))
    for algorithm, entries in fresh.items():
        expected = [{"config": c, "path": p} for c, p in entries]
        # Numerical paths are rebuilt with the production baseline engines;
        # subsequent support, selection and prediction replay is independent.
        assert expected == saved["raw_paths"][algorithm]
        counts["documented_baseline_path_rebuilds"] += len(entries)
    cache = {}
    for method, options in saved["options"].items():
        algorithm, variant = method.split("_", 1)
        for i, model in enumerate(options):
            config, path = fresh[algorithm][i]
            assert model["config"] == config and model["config_index"] == i
            metric = model["metric"]
            assert metric["variant"] == variant
            assert metric["raw_native_support"] == sorted(path["native"]["features"])
            expected = sorted(path["ranking"][:saved["matched_proposed_count"]]) if variant.startswith("matched") else sorted(path["native"]["features"])
            assert metric["features"] == expected and metric["ranking"] == path["ranking"]
            score_replay(A, y, B, truth, model, counts, cache)


def audit(tag):
    threadpool_limits(1)
    artifact = OUT/(tag+".json")
    saved = json.loads(artifact.read_text())
    source_path = ROOT/"results"/saved["source_folder"]/(saved["dataset"]+".json")
    assert saved["fingerprints"]["source"] == sha(source_path)
    assert saved["fingerprints"]["data"] == sha(ROOT/"data"/"processed"/(saved["dataset"]+".npz"))
    source = json.loads(source_path.read_text())
    X, y = load_dataset(saved["dataset"])
    counts = Counter()
    for row in saved["rows"]:
        original = next(r for r in source["rows"] if r["fold"] == row["fold"])
        if row["status"] != "complete":
            assert original.get("status", "complete") != "complete"
            continue
        assert row["train_indices"] == original["train_indices"] and row["test_indices"] == original["test_indices"]
        ri, vi = np.array(row["train_indices"]), np.array(row["test_indices"])
        assert not set(ri) & set(vi)
        raw, labels = X[ri], y[ri]
        chosen = original["fit"]["chosen"]
        inner_count = len(chosen["features"]) if "features" in chosen else chosen["count"]
        for inner, upstream in zip(row["inner"], original["fit"]["inner"][0]["folds"], strict=True):
            assert inner["fit_indices"] == upstream["fit_indices"] and inner["validation_indices"] == upstream["validation_indices"]
            ii, jj = np.array(inner["fit_indices"]), np.array(inner["validation_indices"])
            assert not set(ii) & set(jj) and len(set(ii)|set(jj)) == len(raw)
            A, B = scaling(raw[ii], raw[jj])
            assert inner["matched_proposed_count"] == inner_count
            block_replay(A, labels[ii], B, labels[jj], inner, counts)
            counts["inner_splits"] += 1
        A, B = scaling(raw, X[vi])
        outer = row["outer"]
        assert outer["matched_proposed_count"] == len(original["fit"]["selected"]["features"])
        fresh = json.loads(json.dumps(paths(A, labels)))
        assert outer["raw_paths"] == {a: [{"config": c, "path": p} for c, p in entries] for a, entries in fresh.items()}
        counts["documented_baseline_path_rebuilds"] += sum(len(v) for v in fresh.values())
        cache = {}
        for method, picked in row["selection"].items():
            index = selected_index(row["inner"], method)
            assert index == picked["selected_config_index"]
            result = outer["results"][method]
            if index is None:
                assert result["score"] is None
                continue
            assert result["config_index"] == index
            algorithm, variant = method.split("_", 1)
            config, path = fresh[algorithm][index]
            assert result["config"] == config
            assert result["metric"]["features"] == (sorted(path["ranking"][:outer["matched_proposed_count"]]) if variant.startswith("matched") else sorted(path["native"]["features"]))
            score_replay(A, labels, B, y[vi], result, counts, cache)
            counts["nested_selection_replays"] += 1
        counts["outer_splits"] += 1
        print(saved["dataset"], row["fold"], "independent control audit passed", flush=True)
    report = {"passed": True, "counts": dict(counts), "artifact_sha256": sha(artifact), "auditor_sha256": sha(__file__),
              "goal_achieved": False, "scope": "independent_train_only_scaling_supports_mass_caps_nested_config_choice_raw_weighted_votes_and_endpoint_query_bounds",
              "limitations": ["baseline_numeric_engines_rebuilt_not_independent_author_code_fidelity", "transfer_projection_invariants_checked_not_independent_optimizer_replay", "source_selector_not_reaudited_here", "exposed_development_not_confirmation"]}
    (OUT/(tag+"-audit.json")).write_text(json.dumps(report, indent=2), encoding="utf8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.only), indent=2))
