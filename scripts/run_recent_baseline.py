"""Evaluate a recent FSS comparator on existing, frozen outer splits."""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run(path_string):
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.dummy import DummyClassifier
    from threadpoolctl import threadpool_limits
    from certified_frfs.datasets import load_dataset
    from certified_frfs.recent import select_sfrip
    from certified_frfs.core import certificate
    from certified_frfs.classifiers import FuzzyRoughScoreClassifier
    threadpool_limits(1)
    source = Path(path_string)
    frozen = json.loads(source.read_text(encoding="utf-8"))
    output = ROOT / "results" / "recent-sfrip" / source.name
    fingerprint = hashlib.sha256((ROOT / "src/certified_frfs/recent.py").read_bytes()
                                 + Path(__file__).read_bytes()).hexdigest()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if output.exists():
        cached = json.loads(output.read_text())
        if cached["code_sha256"] != fingerprint or cached["source_sha256"] != source_hash:
            raise ValueError("stale recent-baseline cache")
        return {"split": source.stem, "status": "cached"}
    X, y = load_dataset(frozen["dataset"])
    train, test = frozen["train_indices"], frozen["test_indices"]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = MinMaxScaler(clip=True)
    Xtr = scaler.fit_transform(imputer.fit_transform(X[train]))
    Xte = scaler.transform(imputer.transform(X[test]))
    start = time.perf_counter()
    selection = select_sfrip(Xtr, y[train])
    elapsed = time.perf_counter() - start
    feats = selection["features"]
    weights = np.asarray(frozen["weights"])
    cert = certificate(Xtr, y[train], feats, weights, frozen["eps"])
    rng = np.random.default_rng(frozen["seed"] * 1000 + frozen["fold"])
    perturbations = {0.0: [Xte]}
    for radius in [0.01, 0.05]:
        perturbations[radius] = [np.clip(Xte + rng.uniform(-radius, radius, Xte.shape), 0, 1)
                                 for _ in range(5)]
    rows = []
    models = {"knn": KNeighborsClassifier(n_neighbors=min(5, len(train))),
              "svm": SVC(C=1, gamma="scale"),
              "fuzzy_rough": FuzzyRoughScoreClassifier(weights=weights[feats])}
    for name, model in models.items():
        if not feats:
            # Evaluate the native empty subset as a training-prior constant;
            # do not add an unselected feature to improve the comparator.
            model = DummyClassifier(strategy="prior")
        fit_X = Xtr[:, feats] if feats else np.zeros((len(train), 1))
        model.fit(fit_X, y[train])
        for radius, versions in perturbations.items():
            scores = []
            f1s = []
            for z in versions:
                pred = model.predict(z[:, feats] if feats else np.zeros((len(test), 1)))
                scores.append(balanced_accuracy_score(y[test], pred))
                f1s.append(f1_score(y[test], pred, average="macro", zero_division=0))
            rows.append({"classifier": name, "test_epsilon": radius,
                         "balanced_accuracy": float(np.mean(scores)),
                         "macro_f1": float(np.mean(f1s))})
    result = {"dataset": frozen["dataset"], "seed": frozen["seed"], "fold": frozen["fold"],
              "method": "sfrip_2025_independent", "selection": selection,
              "selection_seconds": elapsed, "d": X.shape[1], "n_features": len(feats),
              "certificate_upper": cert["upper_bound"],
              "meets_our_certificate_budget": bool(cert["upper_bound"] <= frozen["eta"]),
              "rows": rows, "empty_subset": not bool(feats),
              "source_sha256": source_hash, "code_sha256": fingerprint}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {"split": source.stem, "status": "complete", "k": len(feats),
            "d": X.shape[1], "seconds": round(elapsed, 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--first-fold", action="store_true")
    args = ap.parse_args()
    files = sorted((ROOT / "results/primary").glob("*-s*-f*.json"))
    if args.first_fold:
        files = [p for p in files if p.name.endswith("-s17-f0.json")]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, str(p)) for p in files]
        for f in concurrent.futures.as_completed(futures):
            print(json.dumps(f.result()), flush=True)
