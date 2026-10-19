"""Independent recipe, certificate, prediction and shared-noise replay.

The nonlinear relation optimizer and relevance ranking are replayed using the
frozen production functions, explicitly without claiming optimizer independence.
Recipe construction, support resolution, transport/projection, selection, all
guards, votes, query certificates and classwise metrics are separate replays.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.relation_support_selector_v1 import _fit_relation, _ranking
from audit_relation_support_selector_v1 import (
    equivalent, independent_preprocess, independent_evaluate, check_guard,
)
from audit_majority_bound_prediction_v1 import query_certificate

FOLDER = ROOT / "results/selective-support-selector-v3"
FAMILIES = ("uniform_sqrt", "half_majority", "learned_majority")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_recipes(d):
    counts = sorted({max((d + 1) // 2, int(np.ceil(d * f))) for f in (.5, .7, .85, 1.)})
    rows = []
    for k in counts:
        rows.append({"count": k, "remove_rank": None, "add_rank": None})
        for r in range(max(0, k - 2), k):
            for a in range(k, min(d, k + 2)):
                rows.append({"count": k, "remove_rank": r, "add_rank": a})
    return rows


def resolve(ranking, recipe):
    assert sorted(ranking) == list(range(len(ranking)))
    ids = list(ranking[:recipe["count"]])
    if recipe["remove_rank"] is not None:
        assert recipe["remove_rank"] < recipe["count"] <= recipe["add_rank"] < len(ranking)
        ids.remove(ranking[recipe["remove_rank"]])
        ids.append(ranking[recipe["add_rank"]])
    return sorted(ids)


def transported_weights(A, fs, reference):
    """Separate Pearson transfer and capped-simplex scalar bisection."""
    d = A.shape[1]
    if len(fs) == d:
        return reference.copy()
    mass = float(reference.sum())
    v = reference[fs].copy()
    for b in sorted(set(range(d)) - set(fs)):
        scores = []
        xb = A[:, b] - A[:, b].mean()
        for a in fs:
            xa = A[:, a] - A[:, a].mean()
            den = np.sqrt(np.sum(xa * xa) * np.sum(xb * xb))
            scores.append(0. if np.std(A[:, a]) <= 1e-12 or np.std(A[:, b]) <= 1e-12
                          else float(np.sum(xa * xb) / den) ** 2)
        scores = np.asarray(scores)
        if scores.sum() <= 1e-12:
            scores = np.ones(len(fs))
        v += reference[b] * scores / scores.sum()
    z, cap = v / mass, 2. / d
    left, right = float(z.min() - cap), float(z.max())
    for _ in range(100):
        midpoint = (left + right) / 2
        if np.clip(z - midpoint, 0., cap).sum() > 1.:
            left = midpoint
        else:
            right = midpoint
    return np.clip(z - (left + right) / 2, 0., cap) * mass


def fitted_relations(A, y):
    uniform = _fit_relation(A, y, "uniform_sqrt")
    learned = _fit_relation(A, y, "learned_majority")
    return {"uniform_sqrt": uniform, "learned_majority": learned,
            "half_majority": (uniform + learned) / 2}


def check_candidate(A, y, reference, fs, candidate, counts):
    d = A.shape[1]
    assert candidate["features"] == fs
    w = np.asarray(candidate["weights"], float)
    assert w.shape == (len(fs),) and np.isfinite(w).all() and np.all(w >= 0.)
    assert w.max() <= 2. / np.sqrt(d) + 1e-8
    np.testing.assert_allclose(w.sum(), np.sqrt(d), atol=1e-8, rtol=0.)
    equivalent(w.tolist(), transported_weights(A, fs, reference).tolist())
    uniform = np.ones(d) / np.sqrt(d)
    ok1 = check_guard(A, y, fs, uniform, w, candidate["certificate"], candidate["classwise"])
    ok2 = check_guard(A, y, fs, reference, w, candidate["family_certificate"], candidate["family_classwise"])
    assert candidate["joint_feasible"] == bool(ok1 and ok2)
    counts["independent_dual_guard_candidates"] += 1
    counts["infeasible_candidates"] += not candidate["joint_feasible"]


def check_score(A, y, B, truth, model, saved, counts):
    score = independent_evaluate(A, y, B, truth, model["features"], model["weights"])
    for key, value in score.items():
        equivalent(saved[key], value)
    query = query_certificate(A, y, B, np.asarray(score["predictions"]),
                              model["features"], np.asarray(model["weights"]))
    equivalent(saved["query_certificate"], query)
    flags = np.asarray(query["certified"])
    correct = np.asarray(score["predictions"]) == truth
    groups = [truth == label for label in np.unique(truth)]
    measured = {
        "query_certified_BA": float(np.mean([(correct & flags)[g].mean() for g in groups])),
        "query_coverage": float(flags.mean()),
        "balanced_query_coverage": float(np.mean([flags[g].mean() for g in groups])),
        "selective_accuracy": float(correct[flags].mean()) if flags.any() else None,
        "class_certified_counts": [int((flags & g).sum()) for g in groups],
        "class_certified_correct_counts": [int((flags & correct & g).sum()) for g in groups],
        "class_query_counts": [int(g.sum()) for g in groups],
    }
    assert set(saved) == set(score) | set(measured) | {"query_certificate"}
    for key, value in measured.items():
        equivalent(saved[key], value)
    counts["independent_query_scores"] += 1


def check_fit(raw, y, fit, counts):
    d = raw.shape[1]
    assert fit["families"] == list(FAMILIES)
    assert fit["certificate_parameters"] == {"eps": .01, "eta": .1, "retention": .9}
    assert fit["clean_tolerance"] == .005
    grid = canonical_recipes(d)
    equivalent(fit["recipes"], grid)
    splits = list(StratifiedKFold(fit["inner_splits"], shuffle=True, random_state=fit["seed"]).split(raw, y))
    assert len(fit["inner"]) == len(splits)
    for fold, ((ii, jj), block) in enumerate(zip(splits, fit["inner"])):
        assert block["fold"] == fold and block["fit_indices"] == ii.tolist() and block["validation_indices"] == jj.tolist()
        assert set(block["families"]) == set(FAMILIES)
        A, B = preprocess_fit(raw[ii], raw[jj])
        ia, ib = independent_preprocess(raw[ii], raw[jj])
        np.testing.assert_allclose(A, ia, atol=1e-14, rtol=0.)
        np.testing.assert_allclose(B, ib, atol=1e-14, rtol=0.)
        references = fitted_relations(A, y[ii])
        for family in FAMILIES:
            part = block["families"][family]
            equivalent(part["reference_weights"], references[family].tolist())
            assert part["ranking"] == _ranking(A, y[ii], references[family], family, fit["seed"] + fold)
            assert len(part["candidates"]) == len(grid)
            for index, row in enumerate(part["candidates"]):
                assert row["recipe_index"] == index
                fs = resolve(part["ranking"], grid[index])
                check_candidate(A, y[ii], references[family], fs, row["candidate"], counts)
                if row["candidate"]["joint_feasible"]:
                    assert row["score"] is not None
                    check_score(A, y[ii], B, y[jj], row["candidate"], row["score"], counts)
                else:
                    assert row["score"] is None
        print("inner", fold, "all recipe guards and scores replayed", dict(counts), flush=True)
    options = []
    for family in FAMILIES:
        for index, recipe in enumerate(grid):
            candidates = [b["families"][family]["candidates"][index] for b in fit["inner"]]
            if any(c["score"] is None for c in candidates):
                continue
            options.append({"family": family, "recipe_index": index, "recipe": recipe,
                            "mean_BA": float(np.mean([c["score"]["clean_BA"] for c in candidates])),
                            "mean_certified_BA": float(np.mean([c["score"]["query_certified_BA"] for c in candidates])),
                            "mean_coverage": float(np.mean([c["score"]["query_coverage"] for c in candidates]))})
    equivalent(fit["options"], options)
    best = max(c["mean_BA"] for c in options)
    chosen = max((c for c in options if c["mean_BA"] >= best - .005 - 1e-12),
                 key=lambda c: (c["mean_certified_BA"], c["mean_coverage"], c["mean_BA"],
                                -c["recipe"]["count"], -FAMILIES.index(c["family"]), -c["recipe_index"]))
    equivalent(fit["chosen"], chosen)
    A, _ = preprocess_fit(raw, raw)
    reference = fitted_relations(A, y)[chosen["family"]]
    equivalent(fit["refit_reference_weights"], reference.tolist())
    ranking = _ranking(A, y, reference, chosen["family"], fit["seed"] + 100)
    assert fit["refit_ranking"] == ranking
    attempt = fit["refit_attempt"]
    check_candidate(A, y, reference, resolve(ranking, chosen["recipe"]), attempt, counts)
    fallback = not attempt["joint_feasible"]
    selected = fit["selected"]
    assert selected["outer_refit_fallback"] == fallback
    assert selected["requested_count"] == chosen["recipe"]["count"]
    assert selected["realized_count"] == len(selected["features"])
    if fallback:
        uniform = np.ones(d) / np.sqrt(d)
        assert selected["family"] == "uniform_sqrt"
        check_candidate(A, y, uniform, list(range(d)), selected, counts)
        counts["outer_refit_fallbacks"] += 1
    else:
        assert selected["family"] == chosen["family"]
        for key, value in attempt.items():
            equivalent(selected[key], value)
    assert selected["joint_feasible"]
    counts["independent_recipe_selections"] += 1


def audit(name):
    threadpool_limits(1)
    started = time.perf_counter()
    auditor_dependencies = [Path(__file__),
                            ROOT / "scripts/audit_relation_support_selector_v1.py",
                            ROOT / "scripts/audit_majority_bound_prediction_v1.py"]
    audit_hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in auditor_dependencies}
    artifact = FOLDER / (name + ".json")
    artifact_sha = digest(artifact)
    saved = json.loads(artifact.read_text(encoding="utf-8"))
    freeze = json.loads((FOLDER / "source-freeze.json").read_text(encoding="utf-8"))
    assert saved["complete"] and saved["source_hashes"] == freeze["hashes"]
    assert name in freeze["datasets"] and freeze["outer_seed"] == 20261011
    assert (freeze["eps"], freeze["eta"], freeze["retention"]) == (.01, .1, .9)
    for rel, sha in freeze["hashes"].items():
        assert digest(ROOT / rel) == sha, rel
    X, y = load_dataset(name)
    assert saved["dataset"] == name and saved["d"] == X.shape[1]
    counts = Counter()
    splits = list(StratifiedKFold(3, shuffle=True, random_state=20261011).split(X, y))
    assert len(saved["rows"]) == len(splits)
    for fold, ((ii, jj), row) in enumerate(zip(splits, saved["rows"])):
        assert row["fold"] == fold and row["train_indices"] == ii.tolist() and row["test_indices"] == jj.tolist()
        assert row["fit"]["seed"] == 20261011 + fold
        assert row["fit"]["inner_splits"] == min(3, int(np.unique(y[ii], return_counts=True)[1].min()))
        check_fit(X[ii], y[ii], row["fit"], counts)
        A, B = independent_preprocess(X[ii], X[jj])
        d = A.shape[1]
        selected = row["fit"]["selected"]
        models = {"proposed": selected,
                  "full": {"features": list(range(d)), "weights": np.ones(d) / np.sqrt(d)},
                  "matched_uniform": {"features": selected["features"],
                                      "weights": np.ones(len(selected["features"])) / np.sqrt(len(selected["features"]))}}
        assert set(row["shared_perturbations"]) == set(models) == set(row["perturbed_BA"])
        for method, model in models.items():
            check_score(A, y[ii], B, y[jj], model, row[method], counts)
            assert len(row["shared_perturbations"][method]) == 8
        for draw in range(8):
            rng = np.random.default_rng(20262000 + 10 * fold + draw)
            na = np.clip(A + rng.uniform(-.01, .01, A.shape), 0., 1.)
            nb = np.clip(B + rng.uniform(-.01, .01, B.shape), 0., 1.)
            for method, model in models.items():
                score = independent_evaluate(na, y[ii], nb, y[jj], model["features"], model["weights"])
                equivalent(row["shared_perturbations"][method][draw], score)
                flags = np.asarray(row[method]["query_certificate"]["certified"])
                assert not np.any(flags & (np.asarray(score["predictions"]) != row[method]["predictions"]))
                counts["independent_shared_perturbation_predictions"] += 1
        for method in models:
            equivalent(row["perturbed_BA"][method], float(np.mean([s["clean_BA"] for s in row["shared_perturbations"][method]])))
        counts["outer_folds"] += 1
        print(name, fold, "outer audit passed", dict(counts), flush=True)
    assert digest(artifact) == artifact_sha
    for rel, sha in audit_hashes.items():
        assert digest(ROOT / rel) == sha, rel
    for rel, sha in freeze["hashes"].items():
        assert digest(ROOT / rel) == sha, rel
    report = {"passed": True, "dataset": name, "counts": dict(counts),
              "artifact_sha256": artifact_sha, "auditor_sha256": digest(Path(__file__)),
              "audit_dependency_hashes": audit_hashes,
              "freeze_sha256": digest(FOLDER / "source-freeze.json"),
              "seconds": time.perf_counter() - started,
              "limitations": ["relation optimizer and ranking replay reuse frozen production functions",
                              "ordinary floating point with conservative tolerances, not directed rounding",
                              "inner optimizer replay uses production preprocessing to avoid SLSQP rounding divergence",
                              "exposed development pilot; no confirmatory superiority claim"]}
    (FOLDER / (name + "-audit.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.only), indent=2), flush=True)
