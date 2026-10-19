"""Independent audit for the fixed majority-bound prediction study.

The scorer is deliberately treated as untrusted input.  This module rebuilds
the reference/validation table, scalar Manhattan 5NN votes, metrics, query
certificates, and the training-side majority/dependency certificate.  It does
not select a candidate from validation results.  The expected scorer artifact
is one JSON per task under ``results/majority-bound-prediction-v1`` plus a
``complete.json`` summary.  A method record may expose its metric as either
``metric`` or directly as ``features``/``weights`` and its scores as
``evaluation`` or ``final_evaluation``; this keeps the audit compatible with
the fixed scoring protocol while making absent fields fail loudly.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from certified_frfs.datasets import load_dataset  # noqa: E402
from certified_frfs.validated_pool import preprocess_fit  # noqa: E402
from certified_frfs.feasible_transport import weight_certificate  # noqa: E402
from certified_frfs.redistributed_loss import redistributed_loss_bound  # noqa: E402

SEARCH = ROOT / "results" / "majority-bound-search"
FOLDER = ROOT / "results" / "majority-bound-prediction-v1"
CAP = ROOT / "results" / "capped-sparse-extension"
NEWDATA = ROOT / "results" / "newdata-compact"

TASKS = tuple(sorted(p for p in SEARCH.glob("*.json") if not p.name.startswith("complete")))
EPS = 0.01
RETENTION = 0.9
ETA = 0.1
N_DRAWS = 16
DRAW_SEED = 20261060


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equivalent(a, b, atol=2e-9):
    """Recursive numerical equality with an explicit tolerance."""
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a) == set(b), (set(a) ^ set(b))
        for key in a:
            equivalent(a[key], b[key], atol)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        assert len(a) == len(b), (len(a), len(b))
        for x, y in zip(a, b):
            equivalent(x, y, atol)
        return
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        assert bool(a) == bool(b), (a, b)
        return
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        assert np.isclose(float(a), float(b), rtol=0.0, atol=atol), (a, b)
        return
    assert a == b, (a, b)


def source_split(task):
    """Return raw X/y and the frozen reference/validation indices."""
    group, name, split = task["group"], task["dataset"], int(task["split"])
    if group == "original15":
        X, y = load_dataset(name)
        source_path = CAP / f"{name}.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        block = source["splits"][split]
        ri, vi = np.asarray(block["reference_indices"], int), np.asarray(block["validation_indices"], int)
    elif group == "newdata":
        source_path = NEWDATA / f"{name}.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        with np.load(ROOT / "data" / "newdata-compact" / f"{name}.npz") as data:
            X, y = np.asarray(data["X"], float), np.asarray(data["y"], int)
        block = source["splits"][split]
        ri, vi = np.asarray(block["train_indices"], int), np.asarray(block["test_indices"], int)
    else:
        raise AssertionError(f"unknown task group: {group}")
    assert len(set(ri.tolist()).intersection(vi.tolist())) == 0
    assert len(ri) and len(vi) and np.all((ri >= 0) & (ri < len(y))) and np.all((vi >= 0) & (vi < len(y)))
    return X, y, ri, vi, source_path, source, block


def metric_record(record):
    metric = record.get("metric", record)
    assert isinstance(metric, dict) and "features" in metric and "weights" in metric
    fs = [int(v) for v in metric["features"]]
    weights = np.asarray(metric["weights"], float)
    assert fs == sorted(set(fs)) and len(fs) == len(weights) and len(fs) > 0
    assert np.isfinite(weights).all() and np.all(weights >= -1e-12)
    return fs, weights


def independent_evaluate(A, y, B, truth, fs, weights):
    """Scalar distance accumulation and deterministic plurality vote."""
    A, B, y, truth = np.asarray(A, float), np.asarray(B, float), np.asarray(y), np.asarray(truth)
    classes = np.unique(y)
    distance = np.zeros((len(B), len(A)), float)
    for a, w in zip(fs, weights):
        distance += np.abs(B[:, a, None] * w - A[None, :, a] * w)
    nn = np.argsort(distance, axis=1, kind="stable")[:, : min(5, len(A))]
    votes = np.column_stack([(y[nn] == c).sum(axis=1) for c in classes])
    top = votes.max(axis=1)
    pred = np.empty(len(B), dtype=classes.dtype)
    probabilities = (votes + 1.0) / (len(nn[0]) + len(classes))
    for i, row in enumerate(votes):
        contenders = np.flatnonzero(row == top[i])
        if len(contenders) == 1:
            pred[i] = classes[contenders[0]]
            continue
        # Match the documented stable tie rule: mean neighbour distance,
        # then smallest reference index.
        scores = []
        for j in contenders:
            ids = nn[i][y[nn[i]] == classes[j]]
            scores.append((float(np.mean(distance[i, ids])), int(ids.min()), int(j)))
        pred[i] = classes[min(scores)[2]]
    own = np.searchsorted(classes, truth)
    correct = pred == truth
    metrics = {
        "clean_BA": float(np.mean([np.mean(correct[truth == c]) for c in np.unique(truth)])),
        "balanced_logloss": float(np.mean([np.mean(-np.log(probabilities[truth == c, own[truth == c]])) for c in np.unique(truth)])),
        "macro_F1": float(f1_score(truth, pred, average="macro", zero_division=0)),
    }
    return {"predictions": pred.tolist(), "probabilities": probabilities.tolist(),
            "neighbor_indices": nn.tolist(), "metrics": metrics,
            "tied_queries": (np.sum(votes == top[:, None], axis=1) > 1).tolist(),
            "rule": "strict_plurality_then_mean_neighbor_distance_then_reference_id"}


def query_certificate(A, y, B, targets, fs, weights):
    """Independent interval lower bound for fixed nominal 5NN predictions."""
    A, B, y = np.asarray(A, float), np.asarray(B, float), np.asarray(y)
    targets = np.asarray(targets)
    lo_a, hi_a = np.maximum(0., A - EPS), np.minimum(1., A + EPS)
    lo_b, hi_b = np.maximum(0., B - EPS), np.minimum(1., B + EPS)
    distance = np.zeros((len(B), len(A)))
    for a, w in zip(fs, weights):
        distance += np.abs(B[:, a, None] * w - A[None, :, a] * w)
    lower = np.full(len(B), -np.inf)
    for i, target in enumerate(targets):
        friends = np.flatnonzero(y == target)
        enemies = np.flatnonzero(y != target)
        if len(friends) < 3:
            continue
        friends = friends[np.lexsort((friends, distance[i, friends]))][:3]
        if len(enemies) <= 2:
            lower[i] = np.inf
            continue
        total = np.zeros((len(enemies), 3))
        for a, w in zip(fs, weights):
            endpoints = []
            for t in (lo_b[i, a], hi_b[i, a]):
                near = np.maximum(0., np.maximum(lo_a[enemies, a] - t, t - hi_a[enemies, a]))
                far = np.maximum(np.abs(t - lo_a[friends, a]), np.abs(t - hi_a[friends, a]))
                endpoints.append(near[:, None] - far[None, :])
            total += w * np.minimum(endpoints[0], endpoints[1])
        enemy_bounds = total.min(axis=1)
        raw = np.partition(enemy_bounds, 2)[2]
        lower[i] = raw - 1e-10 * (1 + abs(raw))
    flags = lower > 1e-10
    return {"certified": flags.tolist(),
            "lower_bounds": [None if not np.isfinite(v) else float(v) for v in lower],
            "coverage": float(flags.mean()), "k": 5, "q": 3, "allowed_enemies": 2,
            "scope": "fixed_nominal_query_label_shared_reference_and_query_boxes_not_correctness",
            "rounding": "ordinary_float_with_conservative_tolerance_not_directed_rounding"}


def fresh_training_certificate(A, y, fs, weights):
    """Replay majority retention and dependency using the frozen 0.01 box."""
    d = A.shape[1]
    full = np.ones(d) / np.sqrt(d)
    # The source implementation is used only for the interval dependency
    # arithmetic; the majority mask and retention counts are recomputed here.
    def protected(features, w):
        # This is the scalar equivalent of the interval friend/enemy replay
        # used by the independent certificate auditors.  Keep the endpoint,
        # friend, enemy, and feature axes explicit to avoid a hidden tie or
        # reduction-order change.
        out = np.zeros(len(y), bool)
        lo, hi = np.maximum(0., A - EPS), np.minimum(1., A + EPS)
        features = list(features); w = np.asarray(w, float)
        for i in range(len(y)):
            friends = np.flatnonzero((y == y[i]) & (np.arange(len(y)) != i))
            enemies = np.flatnonzero(y != y[i])
            if len(friends) < 3 or len(enemies) <= 2:
                continue
            distances = np.sum(np.abs(A[friends][:, features] - A[i, features]) * w, axis=1)
            friends = friends[np.lexsort((friends, distances))][:3]
            terms = []
            for t in (lo[i, features], hi[i, features]):
                near = np.maximum(0., np.maximum(lo[enemies][:, features] - t, t - hi[enemies][:, features]))
                far = np.maximum(np.abs(t - lo[friends][:, features]), np.abs(t - hi[friends][:, features]))
                terms.append(near[:, None, :] - far[None, :, :])
            values = np.einsum("efd,d->ef", np.minimum(terms[0], terms[1]), w).min(axis=1)
            raw = np.partition(values, 2)[2]
            out[i] = raw - 1e-10 * (1 + abs(raw)) > 1e-10
        return out
    orig = protected(list(range(d)), full)
    reduced = protected(fs, np.asarray(weights, float))
    classes = np.unique(y)
    required = [int(np.ceil(RETENTION * np.sum((y == c) & orig) - 1e-12)) for c in classes]
    retained = [int(np.sum((y == c) & orig & reduced)) for c in classes]
    dep = redistributed_loss_bound(A, y, fs, full, np.asarray(weights, float), eps=EPS)
    independent_upper = independent_dependency_upper(A, y, fs, full, np.asarray(weights, float))
    assert abs(dep["upper_bound"] - independent_upper) <= 2e-9, (dep["upper_bound"], independent_upper)
    return {"feasible": bool(all(a >= b for a, b in zip(retained, required)) and dep["upper_bound"] <= ETA),
            "majority_feasible": bool(all(a >= b for a, b in zip(retained, required))),
            "retained_counts": retained, "required": required,
            "dependency_certificate": dep}


def independent_dependency_upper(A, y, features, full, reduced):
    """Scalar piecewise-box reconstruction independent of the production bound."""
    n, d = A.shape; fs = list(features); omitted = sorted(set(range(d)) - set(fs))
    if len(np.unique(y)) < 2 or (fs == list(range(d)) and np.array_equal(full, reduced)):
        return 0.0
    lo, hi = np.maximum(0., A - EPS), np.minimum(1., A + EPS)
    values = []
    for i in range(n):
        enemies = np.flatnonzero(y != y[i])
        low = {}; high = {}; nominal = {}
        for j in enemies:
            low[j] = sum(float(w) * max(0., float(lo[i,a]-hi[j,a]), float(lo[j,a]-hi[i,a])) for a,w in zip(fs,reduced))
            high[j] = sum(float(w) * max(abs(float(lo[i,a]-hi[j,a])), abs(float(hi[i,a]-lo[j,a]))) for a,w in zip(fs,reduced))
            nominal[j] = sum(abs(float(A[i,a]*full[a]-A[j,a]*full[a])) for a in range(d))
        cap = min(high.values())
        possible = [j for j in enemies if low[j] <= cap + 1e-12]
        witnesses = sorted(enemies, key=lambda j: (nominal[j], int(j)))[:8]
        object_values = []
        for j in possible:
            upper_values = []
            for k in witnesses:
                bound = sum(float(full[a]) * max(abs(float(lo[i,a]-hi[k,a])), abs(float(hi[i,a]-lo[k,a]))) for a in omitted)
                for a, wr in zip(fs, reduced):
                    if j == k:
                        coef = float(full[a]-wr)
                        near = max(0., float(lo[i,a]-hi[j,a]), float(lo[j,a]-hi[i,a]))
                        far = max(abs(float(lo[i,a]-hi[j,a])), abs(float(hi[i,a]-lo[j,a])))
                        bound += coef * (far if coef >= 0 else near)
                    else:
                        points = [float(lo[i,a]), float(hi[i,a])]
                        points += [float(t) for t in (lo[j,a],hi[j,a],lo[k,a],hi[k,a]) if lo[i,a] <= t <= hi[i,a]]
                        bound += max(float(full[a])*max(abs(t-float(lo[k,a])),abs(t-float(hi[k,a]))) - float(wr)*max(0.,float(lo[j,a])-t,t-float(hi[j,a])) for t in points)
                upper_values.append(bound)
            object_values.append(max(0., min(1-low[j], min(upper_values))))
        values.append(max(object_values))
    raw = float(np.mean(values))
    return float(np.clip(raw + 1e-10*(1+abs(raw)), 0., 1.))


def expected_source_metric(task, X, y, source, block, candidate_path):
    """Recover the frozen direct search candidate and its pre-search metric."""
    p = json.loads(candidate_path.read_text(encoding="utf-8"))
    fs = p["fit"]["features"]
    w = p["fit"]["weights"]
    return fs, np.asarray(w, float), p


def check_method(name, record, A, y_ref, B, y_val, counts):
    fs, weights = metric_record(record)
    assert max(fs) < A.shape[1]
    if weights.shape != (len(fs),):
        raise AssertionError(f"{name}: weights must be retained-coordinate vector")
    fresh = independent_evaluate(A, y_ref, B, y_val, fs, weights)
    final = record.get("final_evaluation")
    stored_eval = record.get("evaluation", final)
    if isinstance(stored_eval, dict) and "evaluation" in stored_eval and isinstance(stored_eval["evaluation"], dict):
        final = stored_eval
        stored_eval = stored_eval["evaluation"]
    assert stored_eval is not None, f"{name}: missing evaluation"
    saved_metrics = stored_eval.get("metrics", stored_eval)
    for key, value in fresh["metrics"].items():
        equivalent(value, saved_metrics[key])
    for key in ("predictions", "probabilities", "neighbor_indices", "tied_queries", "rule"):
        if key in stored_eval:
            equivalent(fresh[key], stored_eval[key])
    counts["independent_predictions"] += 1
    cert = fresh_training_certificate(A, y_ref, fs, weights)
    stored_cert = record.get("fresh_certificate", record.get("certificate"))
    if stored_cert is not None:
        # Stored certificates may include provenance-only keys; compare core
        # numerical fields while retaining all failures for diagnosis.
        for key in ("feasible", "majority_feasible", "retained_counts", "required"):
            assert key in stored_cert and stored_cert[key] == cert[key], (name, key)
        assert np.isclose(stored_cert["dependency_certificate"]["upper_bound"],
                          cert["dependency_certificate"]["upper_bound"], atol=2e-9)
    counts["independent_certificates"] += 1
    query = query_certificate(A, y_ref, B, np.asarray(fresh["predictions"]), fs, weights)
    stored_query = record.get("query_certificate", final.get("query_certificate") if isinstance(final, dict) else None)
    if stored_query is not None:
        equivalent(query, stored_query)
    certified_correct = (np.asarray(fresh["predictions"]) == y_val) & np.asarray(query["certified"])
    certified_BA = float(np.mean([np.mean(certified_correct[y_val == c]) for c in np.unique(y_val)]))
    if "query_certified_BA" in saved_metrics:
        equivalent(certified_BA, saved_metrics["query_certified_BA"])
    perturbations = record.get("perturbation_evaluations", record.get("perturbations", []))
    if not perturbations and isinstance(final, dict):
        perturbations = final.get("perturbation_evaluations", final.get("perturbations", []))
    if perturbations:
        assert len(perturbations) == N_DRAWS, (name, len(perturbations))
        values = []
        for draw, stored in enumerate(perturbations):
            rng = np.random.default_rng(DRAW_SEED + draw)
            ref = np.clip(A + rng.uniform(-EPS, EPS, A.shape), 0., 1.)
            qry = np.clip(B + rng.uniform(-EPS, EPS, B.shape), 0., 1.)
            noisy = independent_evaluate(ref, y_ref, qry, y_val, fs, weights)
            expected = stored.get("evaluation", stored)
            equivalent(noisy, expected)
            values.append(noisy["metrics"]["clean_BA"])
            assert not np.any((np.asarray(noisy["predictions"]) != np.asarray(fresh["predictions"])) & np.asarray(query["certified"])), (name, draw, "certificate violation")
            counts["independent_perturbation_predictions"] += 1
        if "mean_perturbed_BA" in saved_metrics:
            assert np.isclose(np.mean(values), saved_metrics["mean_perturbed_BA"], atol=2e-9)
    fresh["metrics"]["query_certified_BA"] = certified_BA
    if perturbations:
        fresh["metrics"]["mean_perturbed_BA"] = float(np.mean(values))
    return {"features": fs, "n_features": len(fs), "metrics": fresh["metrics"],
            "certificate": cert, "query_certificate": query}


def audit_one(path, counts):
    saved = json.loads(path.read_text(encoding="utf-8"))
    task = saved["task"]
    X, y, ri, vi, source_path, source, block = source_split(task)
    assert saved["reference_indices"] == ri.tolist()
    validation_key = "validation_indices"
    assert saved[validation_key] == vi.tolist()
    A, B = preprocess_fit(X[ri], X[vi])
    # Validate the direct search candidate independently from the scorer.
    search = json.loads((SEARCH / path.name).read_text(encoding="utf-8"))
    assert search["task"] == task
    direct_fs, direct_w, _ = expected_source_metric(task, X, y, source, block, SEARCH / path.name)
    methods = saved.get("methods", saved.get("results"))
    assert isinstance(methods, dict) and methods, "missing methods"
    # Require both named modern controls when a prediction artifact claims a
    # baseline comparison.  Aliases are accepted to avoid coupling the audit
    # to presentation-only labels, but an omitted control is reported as an
    # audit failure rather than silently treated as a tie.
    names = set(methods)
    assert names.intersection({"weighted_fgs", "fgs", "fgs120", "fgs_matched"}), "weighted FGS control missing"
    assert names.intersection({"sfrip", "sfrip2025", "sfrip_matched"}), "SFRIP control missing"
    report = {}
    for name, record in methods.items():
        if isinstance(record, dict) and record.get("status") == "missing":
            report[name] = {"status": "missing", "reason": record.get("reason", "")}
            continue
        report[name] = check_method(name, record, A, y[ri], B, y[vi], counts)
    # A method labelled original_candidate must exactly reuse the frozen
    # direct candidate.  This prevents silently scoring a validation-tuned
    # replacement under the baseline name.
    for key in ("direct_candidate", "majority_bound", "majority_search", "majority_raw", "majority_bound_raw"):
        if key in methods:
            fs, w = metric_record(methods[key])
            assert fs == [int(v) for v in direct_fs]
            np.testing.assert_allclose(w, direct_w, rtol=0., atol=2e-9)
    if "original_candidate" in methods:
        original = block["fit"]["candidates"][int(task["candidate"])]
        fs, w = metric_record(methods["original_candidate"])
        assert fs == original["features"]
        np.testing.assert_allclose(w, np.asarray(original["weights"], float)[fs], rtol=0., atol=2e-9)
    assert any(key in methods for key in ("proposed", "majority_bound", "majority_search")), "proposed majority method missing"
    # Provenance hashes are checked if emitted by the scorer.
    hashes = saved.get("source_hashes", saved.get("fingerprints", {}))
    assert hashes, "provenance hashes missing"
    if "source_fingerprints_verified" in saved:
        assert saved["source_fingerprints_verified"] is True
    for rel, sha in hashes.items():
        rel_path = ROOT / rel
        assert rel_path.exists() and digest(rel_path) == sha, rel
    counts["datasets"] += 1
    return {"task": task, "methods": report, "source_path": str(source_path.relative_to(ROOT))}


def main():
    threadpool_limits(1)
    if len(TASKS) != 12:
        raise AssertionError(f"expected 12 frozen majority candidates, found {len(TASKS)}")
    artifacts = sorted(p for p in FOLDER.glob("*.json") if not p.name.startswith("complete"))
    if len(artifacts) != 12:
        raise AssertionError(f"expected 12 prediction artifacts, found {len(artifacts)}")
    counts = Counter()
    reports = [audit_one(path, counts) for path in artifacts]
    summary = json.loads((FOLDER / "complete.json").read_text(encoding="utf-8"))
    assert summary.get("candidates") in (None, 12)
    output = {
        "passed": True,
        "counts": dict(counts),
        "artifact_hashes": {p.stem: digest(p) for p in artifacts},
        "summary_sha256": digest(FOLDER / "complete.json"),
        "auditor_sha256": digest(Path(__file__)),
        "goal_achieved": False,
        "reports": reports,
        "scope": "independent_fixed_split_scalar_5NN_metrics_shared_perturbations_query_and_training_certificate_audit_not_confirmation",
        "baseline_matching": {
            "weighted_fgs": "not assumed matched unless scorer supplies same reference/validation provenance",
            "sfrip": "not assumed matched unless scorer supplies same reference/validation provenance",
        },
    }
    (FOLDER / "complete-audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "reports"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
