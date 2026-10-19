"""Independent nested-split, preprocessing, vote and dual-guard replay.

The engineering trial has no complete source freeze.  This auditor records
that limitation and never promotes it to confirmation or a superiority claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from certified_frfs.datasets import load_dataset
from certified_frfs.relation_support_selector_v1 import _fit_relation, _ranking, _candidate
from certified_frfs.validated_pool import preprocess_fit

FOLDER = ROOT / "results" / "relation-support-selector-v3"


def equivalent(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), (set(actual), set(expected))
        for key in expected:
            equivalent(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            equivalent(a, b)
    elif isinstance(expected, (float, np.floating)):
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-8)
    else:
        assert actual == expected, (actual, expected)


def independent_preprocess(reference, query):
    """Median/min/max fit only on the reference, including empty columns."""
    reference, query = np.asarray(reference, float), np.asarray(query, float)
    medians = np.array([np.median(col[~np.isnan(col)]) if np.any(~np.isnan(col)) else 0.
                        for col in reference.T])
    A = np.where(np.isnan(reference), medians, reference)
    B = np.where(np.isnan(query), medians, query)
    minimum, maximum = A.min(axis=0), A.max(axis=0)
    scale = maximum - minimum
    scale[scale == 0] = 1.
    return np.clip((A-minimum)/scale, 0., 1.), np.clip((B-minimum)/scale, 0., 1.)


def independent_evaluate(A, y, B, truth, features, weights, *, k=5):
    """Scalar distance/votes and direct formulas, no production evaluator."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    y, truth = np.asarray(y), np.asarray(truth)
    fs, w = list(features), np.asarray(weights, float)
    assert len(fs) == len(w) and len(fs) > 0
    assert len(set(fs)) == len(fs) and np.all(w >= 0) and np.isfinite(w).all()
    classes = np.unique(y)
    pred, probs = [], []
    for point in B:
        distances = [sum(float(weight)*abs(float(point[a])-float(row[a]))
                         for a, weight in zip(fs, w)) for row in A]
        nearest = sorted(range(len(y)), key=lambda i: (distances[i], i))[:min(k, len(y))]
        votes = np.array([sum(y[i] == c for i in nearest) for c in classes])
        pred.append(classes[int(np.argmax(votes))])
        probs.append((votes+1.)/(len(nearest)+len(classes)))
    pred, probs = np.asarray(pred), np.asarray(probs)
    truth_classes = np.unique(truth)
    ba = np.mean([np.mean(pred[truth == c] == c) for c in truth_classes])
    f1 = []
    for c in np.union1d(truth, pred):
        tp = np.sum((truth == c) & (pred == c))
        den = 2*tp + np.sum((truth != c) & (pred == c)) + np.sum((truth == c) & (pred != c))
        f1.append(0. if den == 0 else 2*tp/den)
    lookup = {c: i for i, c in enumerate(classes)}
    loss = np.mean([np.mean([-np.log(probs[i, lookup[c]]) for i in np.flatnonzero(truth == c)])
                    for c in truth_classes])
    return {"clean_BA": float(ba), "macro_F1": float(np.mean(f1)),
            "balanced_logloss": float(loss), "predictions": pred.tolist(), "probabilities": probs.tolist()}


def independent_protected(A, y, weights):
    """Exact coordinate endpoint minima with own nominal witness ordering."""
    A, y, weights = np.asarray(A), np.asarray(y), np.asarray(weights)
    lo, hi = np.maximum(0, A-.01), np.minimum(1, A+.01)
    protected = np.zeros(len(A), bool)
    for i in range(len(A)):
        friends = [j for j in range(len(A)) if y[j] == y[i] and j != i]
        if len(friends) < 3:
            continue
        friends = sorted(friends, key=lambda j: (float(np.sum(abs(A[j]-A[i])*weights)), j))[:3]
        enemies = np.flatnonzero(y != y[i])
        if len(enemies) < 3:
            continue
        # The scalar minimum of distance-to-enemy minus distance-to-friend
        # occurs at an anchor interval endpoint.
        terms = []
        for anchor in (lo[i], hi[i]):
            enemy_min = np.maximum(0., np.maximum(lo[enemies]-anchor, anchor-hi[enemies]))
            friend_max = np.maximum(abs(anchor-lo[friends]), abs(anchor-hi[friends]))
            terms.append(enemy_min[:, None, :] - friend_max[None, :, :])
        pair_bounds = np.minimum(*terms) @ weights
        raw = np.sort(pair_bounds.min(axis=1))[2]
        protected[i] = raw-1e-10*(1+abs(raw)) > 1e-10
    return protected


def independent_dependency(A, y, fs, full, reduced):
    """Separate coordinate-breakpoint construction of the witness envelope."""
    A, y, full, reduced = map(np.asarray, (A, y, full, reduced))
    n, d = A.shape
    if len(fs) == d and np.array_equal(full, reduced):
        return np.zeros(n), 0.
    lo, hi = np.maximum(0, A-.01), np.minimum(1, A+.01)
    result = []
    omitted = [a for a in range(d) if a not in fs]
    for i in range(n):
        enemies = np.flatnonzero(y != y[i])
        lows = np.maximum(0., np.maximum(lo[i, fs]-hi[enemies][:, fs], lo[enemies][:, fs]-hi[i, fs])) @ reduced
        highs = np.maximum(abs(lo[i, fs]-hi[enemies][:, fs]), abs(hi[i, fs]-lo[enemies][:, fs])) @ reduced
        possible = enemies[lows <= highs.min()+1e-12]
        possible_low = lows[lows <= highs.min()+1e-12]
        witness = sorted(enemies, key=lambda j: (float(np.sum(abs(A[i]-A[j])*full)), int(j)))[:8]
        costs = np.zeros((len(possible), len(witness)))
        for a in omitted:
            costs += full[a]*np.maximum(abs(lo[i, a]-hi[witness, a]), abs(hi[i, a]-lo[witness, a]))[None, :]
        for q, a in enumerate(fs):
            jl, jh = lo[possible, a, None], hi[possible, a, None]
            kl, kh = lo[witness, a][None, :], hi[witness, a][None, :]
            points = [np.full_like(costs, lo[i, a]), np.full_like(costs, hi[i, a])]
            points += [np.broadcast_to(np.clip(p, lo[i,a], hi[i,a]), costs.shape) for p in (jl, jh, kl, kh)]
            values = [full[a]*np.maximum(abs(t-kl), abs(t-kh))-reduced[q]*np.maximum(0., np.maximum(jl-t, t-jh)) for t in points]
            bound = np.maximum.reduce(values)
            same = possible[:, None] == np.asarray(witness)[None, :]
            coefficient = full[a]-reduced[q]
            near = np.maximum(0., np.maximum(lo[i,a]-jh, jl-hi[i,a]))
            far = np.maximum(abs(lo[i,a]-jh), abs(hi[i,a]-jl))
            bound = np.where(same, coefficient*(far if coefficient >= 0 else near), bound)
            costs += bound
        result.append(float(np.max(np.maximum(0., np.minimum(1-possible_low, costs.min(axis=1))))))
    raw = float(np.mean(result))
    return np.array(result), float(np.clip(raw+1e-10*(1+abs(raw)), 0, 1))


def check_guard(A, y, fs, full, weights, certificate, classwise):
    protected = independent_protected(A, y, full)
    retained = independent_protected(A[:, fs], y, weights)
    labels = np.unique(y)
    groups = [(y == c) & protected for c in labels]
    required = [int(np.ceil(.9*g.sum()-1e-12)) for g in groups]
    counts = [int(np.sum(retained & g)) for g in groups]
    equivalent(certificate["required"], required)
    equivalent(certificate["retained_counts"], counts)
    majority = all(c >= r for c, r in zip(counts, required))
    assert certificate["majority_feasible"] == majority
    values, upper = independent_dependency(A, y, fs, full, weights)
    equivalent(certificate["dependency_certificate"]["upper_bound"], upper)
    if "redistributed_object_bounds" in certificate["dependency_certificate"]:
        equivalent(certificate["dependency_certificate"]["redistributed_object_bounds"], values.tolist())
    assert certificate["feasible"] == (majority and upper <= .1)
    expected_bounds = [float(values[y == c].mean()) for c in labels]
    equivalent(classwise["bounds"], expected_bounds)
    # The production classwise safeguard adds envelope padding unless the
    # original uniform full relation is deployed exactly.
    uniform = np.ones(A.shape[1])/np.sqrt(A.shape[1])
    pad_zero = len(fs) == len(uniform) and np.array_equal(weights, uniform)
    class_ok = all(b+(0 if pad_zero else 1e-10*(1+abs(b))) <= .1 for b in expected_bounds)
    assert classwise["feasible"] == class_ok
    return certificate["feasible"] and class_ok


def check_candidate(A, y, reference, ranking, count, candidate, counts):
    fs = sorted(ranking[:count])
    assert candidate["features"] == fs
    weights = np.asarray(candidate["weights"], float)
    assert weights.shape == (len(fs),) and np.all(weights >= 0)
    np.testing.assert_allclose(weights.sum(), np.sqrt(A.shape[1]), atol=1e-8)
    assert weights.max() <= 2/np.sqrt(A.shape[1])+1e-8
    fresh = _candidate(A, y, reference, ranking, count)
    for key in ("weights", "certificate", "family_certificate", "classwise", "family_classwise", "joint_feasible"):
        equivalent(candidate[key], fresh[key])
    uniform = np.ones(A.shape[1])/np.sqrt(A.shape[1])
    ok1 = check_guard(A, y, fs, uniform, weights, candidate["certificate"], candidate["classwise"])
    ok2 = check_guard(A, y, fs, reference, weights, candidate["family_certificate"], candidate["family_classwise"])
    assert candidate["joint_feasible"] == (ok1 and ok2)
    counts["dual_guard_candidates"] += 1


def audit_fit(raw, y, fit, counts):
    assert fit["certificate_parameters"] == {"eps": .01, "eta": .1, "retention": .9}
    assert fit["families"] == ["uniform_sqrt", "learned_majority"]
    d = raw.shape[1]
    grid = sorted(set(max(int(np.ceil(d/2)), int(np.ceil(d*f))) for f in (.25,.4,.55,.7,.85,1.)))
    assert fit["counts"] == grid
    splits = list(StratifiedKFold(fit["inner_splits"], shuffle=True, random_state=fit["seed"]).split(raw, y))
    choices = []
    for block in fit["inner"]:
        family = block["family"]
        assert len(block["folds"]) == len(splits)
        summaries = []
        for fold, ((ii, jj), row) in enumerate(zip(splits, block["folds"])):
            assert row["fold"] == fold and row["fit_indices"] == ii.tolist() and row["validation_indices"] == jj.tolist()
            assert not set(ii) & set(jj) and len(set(ii)|set(jj)) == len(raw)
            # Reuse the frozen production scaler for optimizer replay.  The
            # outer predictions below still use the independent scalar path;
            # SLSQP is sensitive to 2e-16 min/max rounding in the learned
            # relation fit, so exact optimizer replay is not claimed here.
            A, B = preprocess_fit(raw[ii], raw[jj])
            w = _fit_relation(A, y[ii], family)
            equivalent(row["reference_weights"], w.tolist())
            ranking = _ranking(A, y[ii], w, family, fit["seed"]+fold)
            assert row["ranking"] == ranking and [c["count"] for c in row["candidates"]] == grid
            for candidate in row["candidates"]:
                c = candidate["candidate"]
                check_candidate(A, y[ii], w, ranking, candidate["count"], c, counts)
                if c["joint_feasible"]:
                    assert candidate["score"] is not None
                    equivalent(candidate["score"], independent_evaluate(A, y[ii], B, y[jj], c["features"], c["weights"]))
                    counts["inner_predictions"] += 1
                else:
                    assert candidate["score"] is None
                    counts["infeasible_inner_candidates"] += 1
        for count in grid:
            rows = [c for row in block["folds"] for c in row["candidates"] if c["count"] == count and c["score"] is not None]
            if len(rows) == len(splits):
                summaries.append({"family": family, "count": count, "mean_BA": float(np.mean([r["score"]["clean_BA"] for r in rows])), "mean_logloss": float(np.mean([r["score"]["balanced_logloss"] for r in rows]))})
        equivalent(block["summary"], summaries)
        choices.extend(summaries)
    best = max(c["mean_BA"] for c in choices)
    chosen = min([c for c in choices if c["mean_BA"] >= best-.005-1e-12], key=lambda c: (c["count"],c["mean_logloss"],c["family"]))
    equivalent(fit["chosen"], chosen)
    A, _ = preprocess_fit(raw, raw)
    w = _fit_relation(A, y, chosen["family"])
    equivalent(fit["refit_reference_weights"], w.tolist())
    ranking = _ranking(A, y, w, chosen["family"], fit["seed"]+100)
    assert fit["refit_ranking"] == ranking
    check_candidate(A, y, w, ranking, chosen["count"], fit["refit_attempt"], counts)
    fallback = not fit["refit_attempt"]["joint_feasible"]
    selected = fit["selected"]
    assert selected["outer_refit_fallback"] == fallback and selected["selected_count"] == chosen["count"]
    if fallback:
        uniform = np.ones(d)/np.sqrt(d)
        assert selected["family"] == "uniform_sqrt" and selected["features"] == list(range(d))
        check_candidate(A, y, uniform, list(range(d)), d, selected, counts)
        counts["outer_refit_fallbacks"] += 1
    else:
        assert selected["family"] == chosen["family"]
        check_candidate(A, y, w, ranking, chosen["count"], selected, counts)
    assert selected["joint_feasible"]


def audit(name):
    threadpool_limits(1)
    artifact = FOLDER / (name+".json")
    saved = json.loads(artifact.read_text(encoding="utf-8"))
    X, y = load_dataset(name)
    assert saved["dataset"] == name and saved["d"] == X.shape[1]
    assert saved["fingerprints"]["data"] == hashlib.sha256((ROOT/"data"/"processed"/(name+".npz")).read_bytes()).hexdigest()
    outer_k = min(3, int(np.unique(y, return_counts=True)[1].min()))
    splits = list(StratifiedKFold(outer_k, shuffle=True, random_state=20261011).split(X,y))
    assert len(saved["rows"]) == outer_k
    counts = Counter()
    for fold, ((ii,jj), row) in enumerate(zip(splits,saved["rows"])):
        assert row["fold"] == fold and row["train_indices"] == ii.tolist() and row["test_indices"] == jj.tolist()
        if row.get("status", "complete") != "complete":
            assert row["inner_splits"] < 2
            continue
        assert row["fit"]["seed"] == 20261011+fold
        # fit_outer receives the outer-training table; its nested indices are
        # therefore relative to this table, rather than to the raw dataset.
        raw_train, y_train = X[ii], y[ii]
        audit_fit(raw_train, y_train, row["fit"], counts)
        A,B = preprocess_fit(X[ii],X[jj])
        chosen = row["fit"]["selected"]; fs = chosen["features"]
        models = {"proposed": (fs,chosen["weights"]), "matched_uniform": (fs,np.ones(len(fs))/np.sqrt(len(fs))), "full": (list(range(X.shape[1])),np.ones(X.shape[1])/np.sqrt(X.shape[1]))}
        for method,(features,weights) in models.items():
            equivalent(row[method],independent_evaluate(A,y[ii],B,y[jj],features,weights))
            counts["outer_predictions"] += 1
            values=[]
            for draw in range(8):
                rng=np.random.default_rng(20262000+10*fold+draw)
                Q=np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)
                values.append(independent_evaluate(A,y[ii],Q,y[jj],features,weights)["clean_BA"])
                counts["query_only_perturbation_predictions"] += 1
            key={"proposed":"perturbed_BA","matched_uniform":"matched_perturbed_BA","full":"full_perturbed_BA"}[method]
            equivalent(row[key],float(np.mean(values)))
        print(name,fold,"independent fold audit passed",flush=True)
    report={"passed":True,"dataset":name,"counts":dict(counts),"artifact_sha256":hashlib.sha256(artifact.read_bytes()).hexdigest(),"auditor_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"goal_achieved":False,"scope":"independent_scalar_votes_metrics_train_only_scaling_dual_guard_and_nested_selection_replay","limitations":["engineering_trial_without_complete_source_freeze","reference_learning_and_ranking_replayed_with_production_fit_not_independent_optimizer","query_only_perturbations_not_shared_table_attacks","exposed_development_data_not_confirmatory_SOTA_evidence"]}
    (FOLDER/(name+"-audit.json")).write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--only",required=True);args=parser.parse_args()
    print(json.dumps(audit(args.only),indent=2),flush=True)
