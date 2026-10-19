"""Round-two, post-hoc first-split pilot; never a replacement for primary.

The five exploratory configurations below are fixed before this pilot run.
They reuse seed-17/fold-0 splits, full-table weights, training preprocessing
and the shared training-only MI deletion order. All results, including
unsuccessful deletions and class-specific diagnostics, are retained.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from threadpoolctl import threadpool_limits

from certified_frfs.core import _interval_distances, interval_bounds, positive_region
from certified_frfs.datasets import available_datasets, load_dataset, metadata
from certified_frfs.relative import relative_certificate, select_relative
from certified_frfs.selection import deletion_order


CONFIGURATIONS = [
    {"id": "classwise_q90_e01_independent", "scope": "classwise", "retention": 0.9,
     "eps": 0.01, "method": "independent"},
    {"id": "classwise_q90_e01_hybrid", "scope": "classwise", "retention": 0.9,
     "eps": 0.01, "method": "hybrid"},
    {"id": "global_q90_e01_independent", "scope": "global", "retention": 0.9,
     "eps": 0.01, "method": "independent"},
    {"id": "classwise_q80_e01_independent", "scope": "classwise", "retention": 0.8,
     "eps": 0.01, "method": "independent"},
    {"id": "classwise_q90_e00_independent", "scope": "classwise", "retention": 0.9,
     "eps": 0.0, "method": "independent"},
]
OUTPUT = ROOT / "results/round2/relative-pilot.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    paths = [Path(__file__), *sorted((ROOT / "src").rglob("*.py"))]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def combined_hash(mapping):
    return hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def plain(obj):
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save(result):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(plain(result), indent=2, ensure_ascii=False,
                                allow_nan=False) + "\n", encoding="utf-8")


def evaluate(Xtr, ytr, Xte, yte, features, labels):
    model = KNeighborsClassifier(n_neighbors=min(5, len(ytr)))
    model.fit(Xtr[:, features], ytr)
    prediction = model.predict(Xte[:, features])
    recall = recall_score(yte, prediction, labels=labels, average=None, zero_division=0)
    return {"balanced_accuracy": float(balanced_accuracy_score(yte, prediction)),
            "macro_f1": float(f1_score(yte, prediction, labels=labels,
                                      average="macro", zero_division=0)),
            "per_class_recall": [{"class": int(label), "n_test": int((yte == label).sum()),
                                  "recall": float(value)} for label, value in zip(labels, recall)]}


def lower_memberships(X, y, features, weights, eps):
    lower, upper = interval_bounds(X, eps)
    lo, _ = _interval_distances(lower, upper, np.asarray(features, dtype=int), weights)
    return np.minimum(1.0, np.where(y[:, None] != y[None, :], lo, np.inf).min(axis=1))


def class_diagnostics(cert, Xtr, ytr, features, weights, eps):
    retained_lower = lower_memberships(Xtr, ytr, features, weights, eps)
    rows = []
    for group in cert["groups"]:
        mask = ytr == group["group"]
        rows.append(dict(group,
                         retained_dependency_lower=float(retained_lower[mask].mean()),
                         nominal_retained_over_full=(group["retained_dependency_nominal"] /
                             group["full_dependency_nominal"]
                             if group["full_dependency_nominal"] > 0 else None),
                         nominal_constraint_violated=bool(group["nominal_margin"] > 1e-12)))
    return rows


def enrich_history(selected, Xtr, ytr, weights, full_membership, config):
    features = list(range(Xtr.shape[1]))
    history = []
    groups = ([np.ones(len(ytr), dtype=bool)] if config["scope"] == "global"
              else [ytr == label for label in np.unique(ytr)])
    for original in selected["history"]:
        candidate = [a for a in features if a != original["candidate_removed"]]
        retained = positive_region(Xtr, ytr, candidate, weights)
        signed = config["retention"] * full_membership - retained
        nominal_max = max(float(signed[mask].mean()) for mask in groups)
        if original["accepted"]:
            state = "accepted"
            if nominal_max > 1e-9:
                raise AssertionError("A certified deletion violates its nominal constraint")
        elif nominal_max > 1e-12:
            state = "nominal_violation"
        elif original["max_upper_bound"] > 0:
            state = "certificate_unresolved"
        else:
            state = "numerical_acceptance_margin"
        history.append(dict(original, candidate_features=candidate,
                            nominal_max_margin=nominal_max, diagnostic=state))
        if original["accepted"]:
            features = candidate
    if features != selected["features"]:
        raise AssertionError("Could not reconstruct selector history")
    counts = Counter(item["diagnostic"] for item in history)
    no_deletion = len(features) == Xtr.shape[1]
    reasons = None
    if no_deletion:
        reasons = {
            "nominally_violating_attempts": counts["nominal_violation"],
            "unresolved_upper_bound_attempts": counts["certificate_unresolved"],
            "numerical_acceptance_margin_attempts": counts["numerical_acceptance_margin"],
            "all_attempted_deletions_nominally_violate": bool(history) and
                counts["nominal_violation"] == len(history),
            "interpretation": "An unresolved upper bound is not proof of actual infeasibility.",
        }
    return history, dict(counts), reasons


def run_dataset(name):
    start = time.perf_counter()
    source = ROOT / "results/primary" / f"{name}-s17-f0.json"
    run = json.loads(source.read_text(encoding="utf-8"))
    if run["dataset"] != name or run["seed"] != 17 or run["fold"] != 0:
        raise ValueError(f"Unexpected original run identity: {name}")
    X, y = load_dataset(name)
    meta = metadata(name)
    processed_sha = sha256(ROOT / "data/processed" / f"{name}.npz")
    if processed_sha != meta["processed_sha256"]:
        raise ValueError(f"Processed data checksum mismatch: {name}")
    train, test = np.asarray(run["train_indices"]), np.asarray(run["test_indices"])
    if (len(np.unique(train)) != len(train) or len(np.unique(test)) != len(test)
            or set(train) & set(test) or set(train) | set(test) != set(range(len(X)))):
        raise ValueError(f"Invalid original split: {name}")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = MinMaxScaler(clip=True)
    Xtr = scaler.fit_transform(imputer.fit_transform(X[train]))
    Xte = scaler.transform(imputer.transform(X[test]))
    for current, saved, label in ((scaler.data_min_, run["scaler_min"], "minimum"),
                                  (scaler.data_max_, run["scaler_max"], "maximum"),
                                  (imputer.statistics_, run["imputer_statistics"], "imputation")):
        if not np.allclose(current, saved, rtol=0, atol=1e-14, equal_nan=True):
            raise ValueError(f"Original training preprocessing differs ({label}): {name}")
    ytr, yte = y[train], y[test]
    weights = np.asarray(run["weights"], dtype=float)
    d, labels = Xtr.shape[1], np.unique(ytr)
    if weights.shape != (d,):
        raise ValueError(f"Original full-table weights have invalid shape: {name}")
    order_start = time.perf_counter()
    order = deletion_order(Xtr, ytr, seed=17)
    order_seconds = time.perf_counter() - order_start
    previous_attempts = [step["removed_candidate"] for step in run["deletion_history"]]
    if previous_attempts != [a for a in order if a in previous_attempts]:
        raise ValueError(f"Current MI order differs from original run: {name}")
    full = list(range(d))
    full_membership = positive_region(Xtr, ytr, full, weights)
    full_lower = lower_memberships(Xtr, ytr, full, weights, 0.01)
    full_metrics = evaluate(Xtr, ytr, Xte, yte, full, labels)
    original_full = next(r for r in run["rows"] if r["method"] == "all"
                         and r["classifier"] == "knn" and r["test_epsilon"] == 0)
    if not np.isclose(full_metrics["balanced_accuracy"], original_full["balanced_accuracy"],
                      rtol=0, atol=1e-14):
        raise ValueError(f"Full-table baseline differs from original run: {name}")
    primary_features = run["subsets"]["certified"]["features"]
    primary_relative = relative_certificate(Xtr, ytr, primary_features, weights, 0.01,
                          retention=0.9, scope="classwise", method="independent")
    primary_classwise = class_diagnostics(primary_relative, Xtr, ytr, primary_features,
                                          weights, 0.01)
    output = {
        "dataset": name, "seed": 17, "fold": 0, "n_train": len(train), "n_test": len(test),
        "d": d, "labels": labels, "weights": weights, "mi_deletion_order": order,
        "mi_order_seconds": order_seconds,
        "source_run": source.relative_to(ROOT).as_posix(), "source_run_sha256": sha256(source),
        "source_run_code_fingerprint": run["code_fingerprint"],
        "source_run_protocol_version": run["protocol_version"],
        "data_array_sha256": meta["array_sha256"], "processed_file_sha256": processed_sha,
        "train_indices": train, "test_indices": test,
        "preprocessing_reproduced": True, "original_full_knn_reproduced": True,
        "full_knn_clean": full_metrics,
        "full_positive_region_mean": float(full_membership.mean()),
        "full_positive_region_saturated_fraction": float(np.mean(full_membership >= 1-1e-12)),
        "full_class_diagnostics": [
            {"class": int(label), "n_train": int((ytr == label).sum()),
             "full_dependency_nominal": float(full_membership[ytr == label].mean()),
             "full_dependency_lower_eps01": float(full_lower[ytr == label].mean()),
             "full_positive_region_saturated_fraction": float(np.mean(
                 full_membership[ytr == label] >= 1-1e-12))} for label in labels],
        "primary_absolute_subset_under_new_criterion": {
            "interpretation": "Post-hoc diagnostic only: the original absolute-tolerance selector "
                "was not designed or claimed to satisfy this new relative/classwise criterion.",
            "features": primary_features, "removed_fraction": 1-len(primary_features)/d,
            "original_eps": run["eps"], "original_absolute_eta": run["eta"],
            "relative_classwise_certificate": primary_relative,
            "per_class": primary_classwise,
            "nominally_violating_classes": [r["group"] for r in primary_classwise
                                            if r["nominal_constraint_violated"]],
            "knn_clean": evaluate(Xtr, ytr, Xte, yte, primary_features, labels),
        },
        "configurations": [],
    }
    for config in CONFIGURATIONS:
        kwargs = {key: config[key] for key in ("eps", "retention", "scope", "method")}
        selection_started = time.perf_counter()
        selected = select_relative(Xtr, ytr, weights=weights, order=order, seed=17, **kwargs)
        selection_seconds = time.perf_counter() - selection_started
        features = selected["features"]
        history, counts, no_deletion = enrich_history(selected, Xtr, ytr, weights,
                                                      full_membership, config)
        class_cert = (selected["certificate"] if config["scope"] == "classwise" else
                      relative_certificate(Xtr, ytr, features, weights, config["eps"],
                        retention=config["retention"], scope="classwise", method=config["method"]))
        output["configurations"].append({
            **config, "features": features, "n_features": len(features),
            "removed_fraction": 1-len(features)/d,
            "certificate": selected["certificate"],
            "per_class": class_diagnostics(class_cert, Xtr, ytr, features, weights, config["eps"]),
            "classwise_diagnostic_certified": class_cert["certified"],
            "selection_seconds_excluding_shared_mi_order": selection_seconds,
            "selection_seconds_including_shared_mi_order": selection_seconds + order_seconds,
            "knn_clean": evaluate(Xtr, ytr, Xte, yte, features, labels),
            "history": history, "history_diagnostic_counts": counts,
            "no_deletion_diagnostic": no_deletion,
            "minimality_proven": selected["minimality_proven"],
        })
        print(f"{name}/{config['id']}: retained={len(features)}/{d}, "
              f"upper={selected['certificate']['max_upper_bound']:.6g}, "
              f"selection_seconds={selection_seconds:.2f}", flush=True)
    independent, hybrid = output["configurations"][:2]
    output["hybrid_vs_independent"] = {
        "additional_removed_attributes": independent["n_features"] - hybrid["n_features"],
        "same_subset": independent["features"] == hybrid["features"],
        "same_cardinality": independent["n_features"] == hybrid["n_features"],
    }
    output["seconds"] = time.perf_counter() - start
    return output


def summarize(rows):
    configurations = []
    for config in CONFIGURATIONS:
        selected = [next(c for c in row["configurations"] if c["id"] == config["id"])
                    for row in rows]
        counts = Counter()
        for candidate in selected:
            counts.update(candidate["history_diagnostic_counts"])
        configurations.append({
            **config, "datasets": len(rows),
            "mean_removed_fraction": float(np.mean([r["removed_fraction"] for r in selected])),
            "mean_knn_balanced_accuracy": float(np.mean([r["knn_clean"]["balanced_accuracy"]
                                                        for r in selected])),
            "mean_knn_macro_f1": float(np.mean([r["knn_clean"]["macro_f1"] for r in selected])),
            "zero_deletion_datasets": [row["dataset"] for row, result in zip(rows, selected)
                                       if result["n_features"] == row["d"]],
            "zero_deletion_all_attempts_nominally_violate": [row["dataset"]
                for row, result in zip(rows, selected) if result["no_deletion_diagnostic"]
                and result["no_deletion_diagnostic"]["all_attempted_deletions_nominally_violate"]],
            "zero_deletion_with_unresolved_attempts": [row["dataset"]
                for row, result in zip(rows, selected) if result["no_deletion_diagnostic"]
                and result["no_deletion_diagnostic"]["unresolved_upper_bound_attempts"] > 0],
            "history_diagnostic_counts": dict(counts),
        })
    return {
        "datasets": len(rows), "configurations": configurations,
        "mean_full_knn_balanced_accuracy": float(np.mean([r["full_knn_clean"]["balanced_accuracy"]
                                                         for r in rows])),
        "mean_primary_absolute_removed_fraction": float(np.mean([
            r["primary_absolute_subset_under_new_criterion"]["removed_fraction"] for r in rows])),
        "mean_primary_absolute_knn_balanced_accuracy": float(np.mean([
            r["primary_absolute_subset_under_new_criterion"]["knn_clean"]["balanced_accuracy"]
            for r in rows])),
        "primary_absolute_nominal_relative_violation_datasets": [r["dataset"] for r in rows
            if r["primary_absolute_subset_under_new_criterion"]["nominally_violating_classes"]],
        "primary_absolute_new_classwise_certificate_pass_datasets": [r["dataset"] for r in rows
            if r["primary_absolute_subset_under_new_criterion"]["relative_classwise_certificate"]["certified"]],
        "hybrid_additional_removed_attributes_total": sum(r["hybrid_vs_independent"][
            "additional_removed_attributes"] for r in rows),
        "hybrid_additional_deletion_datasets": [r["dataset"] for r in rows
            if r["hybrid_vs_independent"]["additional_removed_attributes"] > 0],
        "hybrid_changed_subset_datasets": [r["dataset"] for r in rows
            if not r["hybrid_vs_independent"]["same_subset"]],
        "full_zero_lower_reference_classes": [{"dataset": r["dataset"], "class": c["class"]}
            for r in rows for c in r["full_class_diagnostics"]
            if c["full_dependency_lower_eps01"] <= 1e-12],
        "full_positive_region_saturated_fraction_by_dataset": {
            r["dataset"]: r["full_positive_region_saturated_fraction"] for r in rows},
    }


def main():
    started = time.perf_counter()
    hashes = source_hashes()
    names = available_datasets()
    result = {
        "analysis_status": "post_hoc_exploratory_first_split_not_replacement_for_primary",
        "run_status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 17, "fold": 0, "datasets_planned": names,
        "configurations_fixed_before_pilot": CONFIGURATIONS,
        "source_hashes": hashes, "source_fingerprint": combined_hash(hashes),
        "data_manifest_sha256": sha256(ROOT / "data/manifest.json"),
        "threadpool_limit": 1, "classifier": "KNeighborsClassifier(n_neighbors=5), clean held-out rows",
        "preprocessing": "Median imputation and MinMaxScaler(clip=True), fitted only on original training fold",
        "selection_order": "Shared training-only MI order with seed 17, verified against original deletion history",
        "scope_of_guarantee": "Empirical fuzzy positive-region retention for every shared-box table; "
            "not predictive accuracy or population robustness. Zero full dependency can be vacuous.",
        "stability": "interval", "saturation_threshold": 1-1e-12,
        "nominal_violation_diagnostic_tolerance": 1e-12,
        "rows": [],
    }
    save(result)
    try:
        with threadpool_limits(limits=1):
            for name in names:
                result["rows"].append(run_dataset(name))
                result["summary"] = summarize(result["rows"])
                result["seconds"] = time.perf_counter() - started
                save(result)
        final_hashes = source_hashes()
        result["source_hashes_end"] = final_hashes
        result["source_files_unchanged_during_run"] = final_hashes == hashes
        if final_hashes != hashes:
            raise RuntimeError("Source files changed during pilot; results must be rerun before interpretation")
        if sha256(ROOT / "data/manifest.json") != result["data_manifest_sha256"]:
            raise RuntimeError("Dataset manifest changed during pilot")
        if len(result["rows"]) != 15:
            raise RuntimeError("Expected all 15 registered real datasets")
        result["run_status"] = "complete"
        result["seconds"] = time.perf_counter() - started
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        save(result)
        print(json.dumps(result["summary"], indent=2), flush=True)
    except Exception as error:
        result["run_status"] = "failed"
        result["error"] = repr(error)
        result["seconds"] = time.perf_counter() - started
        save(result)
        raise


if __name__ == "__main__":
    main()
