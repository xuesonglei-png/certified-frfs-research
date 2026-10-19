"""Reproducible inference-only timing for frozen primary experiment splits.

The benchmark intentionally excludes selection and model fitting. It measures
the fitted classifier's batch ``predict`` call after selecting the frozen
feature subset, with numerical-library threads restricted to one.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results" / "round4" / "inference-runtime.csv"
SUMMARY = ROOT / "results" / "round4" / "inference-runtime-summary.json"


def load_scaled(record):
    import numpy as np
    from certified_frfs.datasets import load_dataset

    X, y = load_dataset(record["dataset"])
    train = np.asarray(record["train_indices"], dtype=int)
    test = np.asarray(record["test_indices"], dtype=int)
    # Refit the frozen preprocessing on exactly the saved training rows.
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = MinMaxScaler(clip=True)
    Xtr = scaler.fit_transform(imputer.fit_transform(X[train]))
    Xte = scaler.transform(imputer.transform(X[test]))
    return Xtr, y[train], Xte


def timed_predict(model, X):
    """Return a robust batch-predict median with a calibrated repeat count."""
    for _ in range(3):
        model.predict(X)
    calls = 1
    start = time.perf_counter_ns()
    while (time.perf_counter_ns() - start) < 6_000_000 and calls < 512:
        model.predict(X)
        calls += 1
    samples = []
    for _ in range(9):
        start = time.perf_counter_ns()
        for _ in range(calls):
            model.predict(X)
        samples.append((time.perf_counter_ns() - start) / calls)
    samples.sort()
    return samples[len(samples) // 2], calls


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import numpy as np
    from threadpoolctl import threadpool_limits
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from certified_frfs.classifiers import FuzzyRoughScoreClassifier

    records = []
    with threadpool_limits(1):
        for path in sorted((ROOT / "results" / "primary").glob("*-s*-f*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            Xtr, ytr, Xte = load_scaled(record)
            for method in ("all", "certified"):
                features = np.asarray(record["subsets"][method]["features"], dtype=int)
                models = {
                    "knn": KNeighborsClassifier(n_neighbors=min(5, len(ytr))),
                    "svm": SVC(C=1.0, kernel="rbf", gamma="scale"),
                    "fuzzy_rough": FuzzyRoughScoreClassifier(
                        weights=np.asarray(record["weights"])[features]),
                }
                for classifier, model in models.items():
                    model.fit(Xtr[:, features], ytr)
                    ns, calls = timed_predict(model, Xte[:, features])
                    records.append({
                        "dataset": record["dataset"], "seed": record["seed"],
                        "fold": record["fold"], "method": method,
                        "classifier": classifier, "n_test": len(Xte),
                        "n_features": len(features), "d": Xtr.shape[1],
                        "batch_predict_ns": ns, "per_sample_us": ns / len(Xte) / 1_000,
                        "throughput_samples_per_s": len(Xte) * 1e9 / ns,
                        "calibrated_calls": calls,
                    })
            print(f"timed {path.name}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {"protocol": "frozen-primary-inference-v1", "completed_splits": len({
        (r["dataset"], r["seed"], r["fold"]) for r in records}),
        "models": ["knn", "svm", "fuzzy_rough"],
        "methods": ["all", "certified"],
        "timing": "median of 9 calibrated batch-predict measurements; fitting and selection excluded",
        "threads": 1, "platform": platform.platform(), "python": sys.version}
    for classifier in summary["models"]:
        all_rows = [r for r in records if r["classifier"] == classifier and r["method"] == "all"]
        cert_rows = [r for r in records if r["classifier"] == classifier and r["method"] == "certified"]
        ratios = [a["per_sample_us"] / c["per_sample_us"] for a, c in zip(all_rows, cert_rows)]
        summary[classifier] = {
            "all_median_us": float(np.median([r["per_sample_us"] for r in all_rows])),
            "certified_median_us": float(np.median([r["per_sample_us"] for r in cert_rows])),
            "median_speedup": float(np.median(ratios)),
            "mean_speedup": float(np.mean(ratios)),
        }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
