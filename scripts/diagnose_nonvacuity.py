"""Post-hoc diagnostics; never replace the prespecified primary experiment.

Report whether even an empty subset meets the primary absolute tolerance.
Also repeat the same-order selector on the first split using the stricter
absolute tolerance eta = 0.1 * gamma_full(X_nominal). This is NOT a uniform
relative-loss guarantee of the form loss(Z) <= 0.1 * gamma_full(Z).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from threadpoolctl import threadpool_limits

from certified_frfs.core import certificate, gamma
from certified_frfs.datasets import available_datasets, load_dataset, metadata
from certified_frfs.selection import deletion_order, select_certified


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(paths):
    return hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()


def shared_subsets(directory):
    runs = [read_json(path) for path in sorted(directory.glob("*-s*-f*.json"))]
    count, by_dataset = 0, {}
    for run in runs:
        subsets = run["subsets"]
        equal = set(subsets["certified"]["features"]) == set(subsets["mi_matched"]["features"])
        count += int(equal)
        bucket = by_dataset.setdefault(run["dataset"], {"identical": 0, "runs": 0})
        bucket["identical"] += int(equal)
        bucket["runs"] += 1
    return {
        "source_directory": str(directory.relative_to(ROOT)),
        "runs": len(runs), "identical": count,
        "fraction": count / len(runs) if runs else None,
        "by_dataset": by_dataset,
        "protocol_versions": sorted({run.get("protocol_version", "unrecorded-v1") for run in runs}),
        "code_fingerprints": sorted({run.get("code_fingerprint", "unrecorded-v1") for run in runs}),
    }


def diagnose(name, directory, relative_fraction):
    run_path = directory / f"{name}-s17-f0.json"
    run = read_json(run_path)
    X, y = load_dataset(name)
    train, test = np.asarray(run["train_indices"]), np.asarray(run["test_indices"])
    if set(train) & set(test) or set(train) | set(test) != set(range(len(X))):
        raise ValueError(f"Invalid original train/test split for {name}")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = MinMaxScaler(clip=True)
    Xtr = scaler.fit_transform(imputer.fit_transform(X[train]))
    Xte = scaler.transform(imputer.transform(X[test]))
    if not np.allclose(scaler.data_min_, run["scaler_min"], atol=1e-14, rtol=0):
        raise ValueError(f"Original training minimum differs for {name}")
    if not np.allclose(scaler.data_max_, run["scaler_max"], atol=1e-14, rtol=0):
        raise ValueError(f"Original training maximum differs for {name}")
    ytr, yte = y[train], y[test]
    weights = np.asarray(run["weights"], dtype=float)
    eps, eta = run["eps"], run["eta"]
    full = gamma(Xtr, ytr, weights=weights)
    empty_loss = full - gamma(Xtr, ytr, features=[], weights=weights)
    empty_cert = certificate(Xtr, ytr, features=[], weights=weights, eps=eps)
    primary_features = run["subsets"]["certified"]["features"]
    primary_loss = full - gamma(Xtr, ytr, primary_features, weights)
    order = deletion_order(Xtr, ytr, seed=17)
    attempts = [step["removed_candidate"] for step in run["deletion_history"]]
    # The final candidate can be skipped when only one attribute remains.
    if attempts != [a for a in order if a in attempts]:
        raise ValueError(f"Current MI order differs from the saved run for {name}")
    relative_eta = relative_fraction * full
    start = time.perf_counter()
    relative = select_certified(Xtr, ytr, eps=eps, eta=relative_eta,
                                weights=weights, order=order)
    relative_seconds = time.perf_counter() - start
    relative_features = relative["features"]
    model = KNeighborsClassifier(n_neighbors=min(5, len(train)))
    model.fit(Xtr[:, relative_features], ytr)
    pred = model.predict(Xte[:, relative_features])
    primary_knn = next(row for row in run["rows"]
                       if row["method"] == "certified" and row["classifier"] == "knn"
                       and row["test_epsilon"] == 0)
    return {
        "dataset": name, "source_run": str(run_path.relative_to(ROOT)),
        "source_run_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
        "data_array_sha256": metadata(name)["array_sha256"],
        "seed": 17, "fold": 0, "n_train": len(train), "n_test": len(test),
        "d": X.shape[1], "epsilon": eps, "primary_eta": eta,
        "gamma_full": full, "nominal_empty_loss": empty_loss,
        "empty_certificate_upper": empty_cert["upper_bound"],
        "eta_over_gamma_full": eta / full if full > 0 else None,
        "primary_empty_certified": bool(empty_cert["upper_bound"] + 1e-10 <= eta),
        "primary_nominal_empty_feasible": bool(empty_loss <= eta),
        "primary_features": primary_features, "primary_n_features": len(primary_features),
        "primary_nominal_loss": primary_loss,
        "primary_certificate_upper": run["subsets"]["certified"]["certificate"]["upper_bound"],
        "primary_knn_balanced_accuracy": primary_knn["balanced_accuracy"],
        "relative_fraction": relative_fraction, "relative_eta": relative_eta,
        "relative_features": relative_features,
        "relative_n_features": len(relative_features),
        "relative_nominal_loss": relative["certificate"]["nominal_loss"],
        "relative_certificate_upper": relative["certificate"]["upper_bound"],
        "relative_selection_seconds_excluding_shared_order": relative_seconds,
        "relative_knn_balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
        "relative_knn_macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Run directory; default prefers primary if all first splits exist.")
    parser.add_argument("--output", default="results/nonvacuity.json")
    parser.add_argument("--relative-fraction", type=float, default=0.1)
    args = parser.parse_args()
    if not 0 < args.relative_fraction < 1:
        raise ValueError("relative-fraction must be in (0,1)")
    names = available_datasets()
    if args.input:
        directory = ROOT / args.input
    else:
        directory = ROOT / "results" / "primary"
        if not all((directory / f"{name}-s17-f0.json").exists() for name in names):
            directory = ROOT / "results" / "archive" / "primary-v1"
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        rows = []
        for name in names:
            row = diagnose(name, directory, args.relative_fraction)
            rows.append(row)
            print(f"{name}: empty_certified={row['primary_empty_certified']}, "
                  f"features {row['primary_n_features']} -> {row['relative_n_features']}", flush=True)
    overlap = {}
    for label, path in (("archived_v1", ROOT / "results/archive/primary-v1"),
                        ("current_primary", ROOT / "results/primary")):
        if path.exists():
            overlap[label] = shared_subsets(path)
    result = {
        "analysis_status": "post_hoc_exploratory_not_a_replacement_for_primary",
        "interpretation": "eta_rel is an absolute threshold scaled by nominal full dependency; "
                          "it does not certify a 10% relative loss at every perturbed table.",
        "source_directory": str(directory.relative_to(ROOT)),
        "relative_fraction": args.relative_fraction,
        "diagnostic_script_sha256": fingerprint([Path(__file__)]),
        "current_src_fingerprint": fingerprint(sorted((ROOT / "src/certified_frfs").glob("*.py"))),
        "data_manifest_sha256": fingerprint([ROOT / "data/manifest.json"]),
        "rows": rows, "subset_overlap": overlap,
        "summary": {
            "datasets": len(rows),
            "empty_certified_datasets": [r["dataset"] for r in rows if r["primary_empty_certified"]],
            "nominal_empty_feasible_datasets": [r["dataset"] for r in rows if r["primary_nominal_empty_feasible"]],
            "median_eta_over_gamma_full": float(np.median([r["eta_over_gamma_full"] for r in rows
                                                           if r["eta_over_gamma_full"] is not None])),
            "primary_mean_removed_fraction": float(np.mean([1-r["primary_n_features"]/r["d"] for r in rows])),
            "relative_mean_removed_fraction": float(np.mean([1-r["relative_n_features"]/r["d"] for r in rows])),
            "relative_zero_removal_datasets": [r["dataset"] for r in rows if r["relative_n_features"] == r["d"]],
            "primary_mean_knn_balanced_accuracy": float(np.mean([r["primary_knn_balanced_accuracy"] for r in rows])),
            "relative_mean_knn_balanced_accuracy": float(np.mean([r["relative_knn_balanced_accuracy"] for r in rows])),
        },
        "seconds": time.perf_counter() - started,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
