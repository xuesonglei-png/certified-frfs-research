"""Independent endpoint/third-order replay of training-only admissions.

No outer predictions, test labels, optimization, or fitted weights are produced.
Dependency bounds are preserved artifact inputs; their mathematical replay is
outside this audit and remains the responsibility of the frozen v4 auditor.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit

FOLDER = ROOT/"results"/"expanded-majority-admission-v1"
SOURCE = ROOT/"results"/"repair-relation-selector-v4"
NAMES = ("banknote", "glass", "heart_statlog", "ionosphere", "seeds", "wdbc")


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def family_reference(candidate):
    dep = candidate["family_certificate"]["dependency_certificate"]
    return np.asarray(dep.get("full_weights", candidate["weights"]))


def scalar_coordinate_endpoints(A, y, w):
    """Own coordinate minima; no production triplet/certificate functions."""
    A, y, w = np.asarray(A), np.asarray(y), np.asarray(w)
    lo, hi = np.maximum(0., A-.01), np.minimum(1., A+.01)
    before = A*w
    distances = np.zeros((len(A), len(A)))
    for a in range(A.shape[1]):
        distances += abs(before[:, a, None]-before[None, :, a])
    small = np.full(len(A), -np.inf)
    large = np.full(len(A), -np.inf)
    for i, label in enumerate(y):
        friends = np.flatnonzero((y == label) & (np.arange(len(y)) != i))
        enemies = np.flatnonzero(y != label)
        if len(friends) < 3:
            continue
        if len(enemies) <= 2:
            small[i] = large[i] = np.inf
            continue
        friends = friends[np.lexsort((friends, distances[i, friends]))][:15]
        values = np.zeros((len(enemies), len(friends)))
        for a in range(A.shape[1]):
            minima = []
            for anchor in (lo[i, a], hi[i, a]):
                enemy_distance = np.maximum(0., np.maximum(lo[enemies,a]-anchor, anchor-hi[enemies,a]))
                friend_distance = np.maximum(abs(anchor-lo[friends,a]), abs(anchor-hi[friends,a]))
                minima.append(enemy_distance[:, None]-friend_distance[None, :])
            values += w[a]*np.minimum(minima[0], minima[1])
        for target, table in ((small, values[:, :3]), (large, values)):
            third_friend = np.sort(table, axis=1)[:, -3]
            raw = np.sort(third_friend)[2]
            target[i] = raw-1e-10*(1+abs(raw))
    return small, large


def counts(flags, protected, y):
    return [int(np.sum(flags & protected & (y == label))) for label in np.unique(y)]


def check_nominal_soundness(A, y, w, flags):
    """Independent LOO votes: checks implications on the training table only."""
    D = np.zeros((len(A), len(A)))
    for a in range(A.shape[1]):
        D += w[a]*abs(A[:,a,None]-A[None,:,a])
    np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, axis=1, kind="stable")[:, :min(5, len(A)-1)]
    labels = np.unique(y)
    votes = np.column_stack([np.sum(y[nn] == label, axis=1) for label in labels])
    pred = labels[votes.argmax(axis=1)]
    assert not np.any(flags & (pred != y))
    return int(flags.sum())


def audit(name):
    threadpool_limits(1)
    source_path, diagnostic_path = SOURCE/(name+".json"), FOLDER/(name+".json")
    saved = json.loads(source_path.read_text(encoding="utf-8"))
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    assert saved["complete"] and diagnostic["source_sha256"] == digest(source_path)
    assert not diagnostic["new_outer_scoring"]
    expected = {(r["outer_fold"], r["inner_fold"], r["family"], tuple(r["features"])): r for r in diagnostic["rows"]}
    seen, tally, details, selected_details = set(), Counter(), [], []
    X, y = load_dataset(name)
    for outer in saved["rows"]:
        raw, labels = X[outer["train_indices"]], y[outer["train_indices"]]
        for inner_fold, block in enumerate(outer["fit"]["inner"]):
            ii = block["fit_indices"]
            A, _ = preprocess_fit(raw[ii], raw[ii])
            truth = labels[ii]
            uniform = np.ones(A.shape[1])/np.sqrt(A.shape[1])
            uniform_protected = scalar_coordinate_endpoints(A, truth, uniform)[0] > 1e-10
            for family, part in block["families"].items():
                reference = np.asarray(part["reference_weights"])
                family_protected = scalar_coordinate_endpoints(A, truth, reference)[0] > 1e-10
                unique = {(tuple(c["features"]), tuple(c["weights"]), tuple(family_reference(c))): c for row in part["candidates"] for c in row["trace"]}
                for (fs, _, _), candidate in unique.items():
                    weights = np.asarray(candidate["weights"])
                    small, large = scalar_coordinate_endpoints(A[:,fs], truth, weights)
                    assert np.all(large >= small-1e-12)
                    flags, original_flags = large > 1e-10, small > 1e-10
                    assert np.all(~original_flags | flags)
                    tally["candidate_flag_monotonicity_checks"] += 1
                    tally["nominal_certified_training_votes_checked"] += check_nominal_soundness(A[:,fs], truth, weights, flags)
                    candidate_family_reference = family_reference(candidate)
                    protection_family = family_protected if np.array_equal(candidate_family_reference, reference) else scalar_coordinate_endpoints(A, truth, candidate_family_reference)[0] > 1e-10
                    protections = ((uniform_protected, "certificate"), (protection_family, "family_certificate"))
                    for protection, guard_name in protections:
                        required = [int(np.ceil(.9*np.sum(protection & (truth == label))-1e-12)) for label in np.unique(truth)]
                        assert candidate[guard_name]["required"] == required
                        assert candidate[guard_name]["retained_counts"] == counts(original_flags, protection, truth)
                    dependency_ok = all(candidate[k]["dependency_certificate"]["upper_bound"] <= .1 for k in ("certificate", "family_certificate"))
                    classwise_ok = candidate["classwise"]["feasible"] and candidate["family_classwise"]["feasible"]
                    majority_ok = all(candidate[k]["majority_feasible"] for k in ("certificate", "family_certificate"))
                    assert candidate["joint_feasible"] == bool(dependency_ok and classwise_ok and majority_ok)
                    tally["all_unique_inner_candidates"] += 1
                    tally["dependency_global_failed"] += not dependency_ok
                    tally["dependency_classwise_failed"] += not classwise_ok
                    tally["majority_failed"] += not majority_ok
                    if candidate["joint_feasible"]:
                        tally["legacy_admitted"] += 1
                    elif not (dependency_ok and classwise_ok):
                        tally["rejected_by_dependency_or_classwise"] += 1
                    else:
                        tally["rejected_only_by_majority"] += 1
                        key = (outer["fold"], inner_fold, family, fs)
                        assert key in expected
                        row = expected[key]
                        seen.add(key)
                        retained, accepted = [], []
                        for protection, guard_name in protections:
                            required = [int(np.ceil(.9*np.sum(protection & (truth == label))-1e-12)) for label in np.unique(truth)]
                            assert candidate[guard_name]["required"] == required
                            assert candidate[guard_name]["retained_counts"] == counts(original_flags, protection, truth)
                            retained.append(counts(flags, protection, truth))
                            accepted.append(all(c >= r for c, r in zip(retained[-1], required)))
                        assert row["retained"] == retained
                        assert row["newly_admitted"] == all(accepted)
                        additional = int(np.sum(flags & ~original_flags))
                        assert row["additional_correct_training_certificates"] == additional
                        tally["expanded_newly_admitted"] += all(accepted)
                        tally["expanded_still_majority_rejected"] += not all(accepted)
                        tally["additional_training_certificates"] += additional
                        details.append({**row, "required": [candidate[k]["required"] for k in ("certificate", "family_certificate")],
                                        "flag_sha256": hashlib.sha256(flags.tobytes()).hexdigest(), "monotonic_flags_passed": True, "nominal_soundness_passed": True})
            print(name, outer["fold"], inner_fold, "training admission audit passed", flush=True)
        A, _ = preprocess_fit(raw, raw)
        d = A.shape[1]
        selected = outer["fit"]["selected"]
        fs, weights = selected["features"], np.asarray(selected["weights"])
        legacy_flags, expanded_flags = scalar_coordinate_endpoints(A[:,fs], labels, weights)
        legacy_flags, expanded_flags = legacy_flags > 1e-10, expanded_flags > 1e-10
        assert np.all(~legacy_flags | expanded_flags)
        selected_counts = {}
        for guard_name, reference in (("certificate", np.ones(d)/np.sqrt(d)), ("family_certificate", family_reference(selected))):
            protected = scalar_coordinate_endpoints(A, labels, reference)[0] > 1e-10
            required = [int(np.ceil(.9*np.sum(protected & (labels == c))-1e-12)) for c in np.unique(labels)]
            assert selected[guard_name]["required"] == required
            retained = counts(legacy_flags, protected, labels)
            assert selected[guard_name]["retained_counts"] == retained
            selected_counts[guard_name] = {"required": required, "legacy_retained": retained, "expanded_retained": counts(expanded_flags, protected, labels)}
        tally["selected_candidates_replayed"] += 1
        tally["nominal_certified_training_votes_checked"] += check_nominal_soundness(A[:,fs], labels, weights, expanded_flags)
        selected_details.append({"outer_fold": outer["fold"], "features": fs, "retained_counts": selected_counts})
    assert seen == set(expected)
    assert tally["expanded_newly_admitted"] == diagnostic["newly_admitted"]
    assert tally["additional_training_certificates"] == diagnostic["additional_certificates"]
    report = {"passed": True, "dataset": name, "counts": dict(tally), "diagnostic_candidates": details,
              "selected_training_candidates": selected_details, "source_sha256": digest(source_path),
              "diagnostic_sha256": digest(diagnostic_path), "auditor_sha256": digest(__file__),
              "no_new_outer_scoring": True, "no_independent_optimizer_claim": True,
              "scope": "independent_coordinate_endpoint_majority_guard_training_only",
              "limitations": ["dependency_values_preserved_not_rederived_here", "preprocessing_reused_from_frozen_production", "floating_point_not_formally_verified"]}
    (FOLDER/(name+"-audit-v1.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(name, dict(tally), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=NAMES)
    args = parser.parse_args()
    for name in ((args.only,) if args.only else NAMES):
        audit(name)
