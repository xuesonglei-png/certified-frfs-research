"""Fixed-budget direct candidate certification, using training information only."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.selection import deletion_order
from certified_frfs.normalized_relative import normalized_relative_certificate

INPUT = ROOT / "results/round2/normalized-pilot.json"
OUTPUT = ROOT / "results/round2/normalized-candidate-pilot.json"
NOTE = ROOT / "research/round2/normalized-candidate-pilot.md"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes():
    return {str(p.relative_to(ROOT)): digest(p) for p in
            [Path(__file__), *sorted((ROOT / "src/certified_frfs").glob("*.py"))]}


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def metrics(model, table, y, labels, features):
    pred = model.predict(table[:, features])
    return {"balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "per_class_recall": recall_score(y, pred, labels=labels, average=None,
                                            zero_division=0).tolist()}


def run_dataset(old, deadline):
    started = perf_counter()
    name = old["dataset"]
    primary_path = ROOT / "results/primary" / f"{name}-s17-f0.json"
    data_path = ROOT / "data/processed" / f"{name}.npz"
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    assert digest(primary_path) == old["source_run_sha256"]
    assert digest(data_path) == old["processed_data_sha256"]
    X, y = load_dataset(name)
    train, test = np.asarray(old["train_indices"]), np.asarray(old["test_indices"])
    np.testing.assert_array_equal(train, primary["train_indices"])
    np.testing.assert_array_equal(test, primary["test_indices"])
    assert not set(train) & set(test)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = MinMaxScaler(clip=True)
    Xtr = scaler.fit_transform(imputer.fit_transform(X[train]))
    Xte = scaler.transform(imputer.transform(X[test]))
    np.testing.assert_allclose(imputer.statistics_, primary["imputer_statistics"], atol=1e-14, rtol=0)
    np.testing.assert_allclose(scaler.data_min_, primary["scaler_min"], atol=1e-14, rtol=0)
    np.testing.assert_allclose(scaler.data_max_, primary["scaler_max"], atol=1e-14, rtol=0)
    ytr, yte = y[train], y[test]
    order = deletion_order(Xtr, ytr, seed=17)
    assert order == old["mi_deletion_order"]
    weights, labels, d = np.asarray(old["weights"]), np.unique(ytr), Xtr.shape[1]
    original = next(c for c in old["configurations"] if c["id"] == "normalized_class_q90_e01_hybrid")
    assert original["certificate"]["certified"]
    pool = {}

    def add(features, source):
        pool.setdefault(tuple(sorted(features)), []).append(source)

    add(original["features"], "original_greedy")
    add(range(d), "full")
    for k in range(d):
        add([k], f"singleton_{k}")
    for k in range(1, d):
        add(order[-k:], f"mi_top_{k}")
    candidates = []
    for features, origins in pool.items():
        if perf_counter() > deadline - 4:
            break
        cert_start = perf_counter()
        cert = normalized_relative_certificate(Xtr, ytr, features, weights, eps=.01,
                                               retention=.9, scope="classwise",
                                               method="hybrid", stability="interval")
        candidates.append({"features": list(features), "sources": origins,
                           "n_features": len(features), "certificate": cert,
                           "seconds": perf_counter() - cert_start})
    result = {"dataset": name, "d": d, "seed": 17, "fold": 0, "n_train": len(train),
              "n_test": len(test), "train_indices": train, "test_indices": test,
              "labels": labels, "weights": weights, "mi_deletion_order": order,
              "source_run_sha256": digest(primary_path), "processed_data_sha256": digest(data_path),
              "scaler_min": scaler.data_min_, "scaler_max": scaler.data_max_,
              "imputer_statistics": imputer.statistics_,
              "candidate_pool": [{"features": list(fs), "sources": origins} for fs, origins in pool.items()],
              "candidate_count": len(pool), "candidates_evaluated": len(candidates),
              "candidates": candidates, "candidate_pool_complete": len(candidates) == len(pool),
              "original_greedy_features": original["features"],
              "original_greedy_evaluations": [r for r in old["evaluations"] if r["method"] == "normalized_hybrid"]}
    if len(candidates) < len(pool):
        result.update(status="partial_candidate_pool", seconds=perf_counter()-started)
        return result
    eligible = [row for row in candidates if row["certificate"]["certified"]]
    selected = min(eligible, key=lambda row: (row["n_features"], row["certificate"]["max_upper_bound"], row["features"]))
    assert selected["n_features"] <= len(original["features"])
    picked = selected["features"]
    matched = sorted(order[-len(picked):])
    result.update(selected_features=picked, selected_sources=selected["sources"],
                  selected_certificate=selected["certificate"], matched_mi_features=matched,
                  additional_deleted_vs_greedy=len(original["features"])-len(picked),
                  changed_subset_vs_greedy=picked != original["features"],
                  candidate_equals_matched_mi=picked == matched,
                  removed_fraction=1-len(picked)/d,
                  original_greedy_removed_fraction=1-len(original["features"])/d)
    predictors = {"all": list(range(d)), "candidate": picked, "mi_matched_candidate": matched}
    models = {method: KNeighborsClassifier(n_neighbors=5).fit(Xtr[:, fs], ytr)
              for method, fs in predictors.items()}
    base_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    rng = np.random.default_rng(base_seed+17)
    evaluation = []
    for noise in (0., .01, .05):
        for repetition in range(1 if noise == 0 else 5):
            table = Xte if noise == 0 else np.clip(Xte+rng.uniform(-noise, noise, Xte.shape), 0., 1.)
            for method, features in predictors.items():
                row = {"method": method, "test_epsilon": noise, "repetition": repetition,
                       **metrics(models[method], table, yte, labels, features)}
                evaluation.append(row)
                if method == "all":
                    former = next(e for e in old["evaluations"] if e["method"] == "all" and
                                  e["test_epsilon"] == noise and e["repetition"] == repetition)
                    assert np.isclose(row["balanced_accuracy"], former["balanced_accuracy"], atol=1e-14, rtol=0)
                    np.testing.assert_allclose(row["per_class_recall"], former["per_class_recall"], atol=1e-14, rtol=0)
    assert digest(primary_path) == result["source_run_sha256"]
    assert digest(data_path) == result["processed_data_sha256"]
    result.update(evaluations=evaluation, status="complete", seconds=perf_counter()-started)
    print(json.dumps({"dataset": name, "greedy": len(original["features"]),
                      "candidate": len(picked), "candidate_count": len(pool),
                      "sources": selected["sources"], "seconds": result["seconds"]}), flush=True)
    return result


def summary(rows):
    complete = [r for r in rows if r["status"] == "complete"]
    out = {"completed_datasets": len(complete),
           "candidate_count": sum(r["candidate_count"] for r in rows),
           "certificates_evaluated": sum(r["candidates_evaluated"] for r in rows)}
    if not complete:
        return out
    out.update(mean_removed_fraction=float(np.mean([r["removed_fraction"] for r in complete])),
               original_greedy_mean_removed_fraction=float(np.mean([r["original_greedy_removed_fraction"] for r in complete])),
               total_additional_deleted_vs_greedy=sum(r["additional_deleted_vs_greedy"] for r in complete),
               additional_deletion_datasets=[r["dataset"] for r in complete if r["additional_deleted_vs_greedy"]],
               changed_subset_datasets=[r["dataset"] for r in complete if r["changed_subset_vs_greedy"]],
               candidate_equals_matched_mi=sum(r["candidate_equals_matched_mi"] for r in complete))
    out["predictive"] = []
    for method in ("all", "candidate", "mi_matched_candidate", "original_greedy"):
        for noise in (0., .01, .05):
            per_dataset = []
            for row in complete:
                evaluations = row["original_greedy_evaluations"] if method == "original_greedy" else row["evaluations"]
                key = "normalized_hybrid" if method == "original_greedy" else method
                per_dataset.append(np.mean([e["balanced_accuracy"] for e in evaluations
                                            if e["method"] == key and e["test_epsilon"] == noise]))
            out["predictive"].append({"method": method, "test_epsilon": noise,
                                      "mean_balanced_accuracy": float(np.mean(per_dataset))})
    return out


def main():
    start = perf_counter()
    deadline = start + 120
    before = hashes()
    input_before, protocol_before = digest(INPUT), digest(NOTE)
    old = json.loads(INPUT.read_text(encoding="utf-8"))
    assert old["run_status"] == "complete" and len(old["rows"]) == 15
    for path, value in old["source_hashes_end"].items():
        if path.startswith("src"):
            assert digest(ROOT / path) == value, "Frozen source differs from reference pilot"
    result = {"protocol": "normalized-fixed-pool-direct-certification-first-fold-2026-09-20-v1",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "analysis_status": "post_hoc_search_exploration_same_normalized_target",
              "retention": .9, "eps": .01, "scope": "classwise", "method": "hybrid", "stability": "interval",
              "pool_rule": "original greedy + full + all singletons + MI top-k for k=1..d-1; deduplicate",
              "selection_rule": "certified candidates: minimize (cardinality, max_upper_bound, lexicographic feature list)",
              "test_use": "evaluation only, never used for subset choice or parameter tuning",
              "runtime_budget_seconds": 120, "reference_pilot_sha256_before": input_before,
              "protocol_note_sha256_before": protocol_before, "source_hashes_before": before,
              "data_manifest_sha256_before": digest(ROOT / "data/manifest.json"), "rows": []}
    try:
        with threadpool_limits(limits=1):
            for row in old["rows"]:
                if perf_counter() > deadline - 4:
                    break
                result["rows"].append(run_dataset(row, deadline))
                result["run_status"] = "running"
                OUTPUT.write_text(json.dumps(result, indent=2, default=plain, allow_nan=False), encoding="utf-8")
                if result["rows"][-1]["status"] != "complete":
                    break
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result.update(seconds=perf_counter()-start, source_hashes_after=hashes(),
                  reference_pilot_sha256_after=digest(INPUT), protocol_note_sha256_after=digest(NOTE),
                  data_manifest_sha256_after=digest(ROOT / "data/manifest.json"),
                  summary=summary(result["rows"]))
    result["valid_frozen_run"] = (before == result["source_hashes_after"] and
                                   input_before == result["reference_pilot_sha256_after"] and
                                   protocol_before == result["protocol_note_sha256_after"] and
                                   result["data_manifest_sha256_before"] == result["data_manifest_sha256_after"])
    result["run_status"] = ("failed" if "error" in result else
                             "complete" if len(result["rows"]) == 15 and
                             all(r["status"] == "complete" for r in result["rows"]) and result["valid_frozen_run"]
                             else "partial")
    OUTPUT.write_text(json.dumps(result, indent=2, default=plain, allow_nan=False), encoding="utf-8")
    print(json.dumps({"run_status": result["run_status"], "seconds": result["seconds"],
                      "valid_frozen_run": result["valid_frozen_run"], "summary": result["summary"]}), flush=True)


if __name__ == "__main__":
    main()
