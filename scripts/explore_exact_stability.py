"""Exploratory stronger stability test; never overwrites primary results.

Compare identical first-fold subsets selected by the default interval
certificate. This isolates certificate strength, not classifier accuracy
or a claim about a prespecified primary experiment.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run_dataset(job):
    name, source, seed, eps, eta, reselect = job
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import MinMaxScaler
    from certified_frfs.core import certificate
    from certified_frfs.datasets import load_dataset
    from certified_frfs.selection import deletion_order, select_certified

    X, y = load_dataset(name)
    splits = min(3, int(np.unique(y, return_counts=True)[1].min()))
    train, _ = next(StratifiedKFold(splits, shuffle=True, random_state=seed).split(X, y))
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = MinMaxScaler(clip=True)
    Xtr = scaler.fit_transform(imputer.fit_transform(X[train]))
    ytr = y[train]
    weights = np.full(Xtr.shape[1], 1 / np.sqrt(Xtr.shape[1]))
    source_path = ROOT / source / f"{name}-s{seed}-f0.json"
    if source_path.exists():
        original = json.loads(source_path.read_text(encoding="utf-8"))
        if original["eps"] != eps or original["eta"] != eta or original["train_indices"] != train.tolist():
            raise ValueError(f"Incompatible source fold: {source_path}")
        features = original["subsets"]["certified"]["features"]
        source_description = str(source_path.relative_to(ROOT))
    else:
        selected = select_certified(Xtr, ytr, weights=weights, eps=eps, eta=eta, seed=seed)
        features = selected["features"]
        source_description = "recomputed_default_interval_subset"
    start = time.perf_counter()
    interval = certificate(Xtr, ytr, features, weights, eps, stability="interval")
    interval_seconds = time.perf_counter() - start
    start = time.perf_counter()
    exact = certificate(Xtr, ytr, features, weights, eps, stability="exact")
    exact_seconds = time.perf_counter() - start
    assert exact["stable_count"] >= interval["stable_count"]
    assert exact["upper_bound"] <= interval["upper_bound"] + 1e-10
    row = dict(dataset=name, seed=seed, fold=0, eps=eps, eta=eta,
                n_train=len(train), d=Xtr.shape[1], features=features,
                n_features=len(features), subset_source=source_description,
                interval=interval, exact=exact, interval_seconds=interval_seconds,
                exact_seconds=exact_seconds,
                stable_fraction_gain=exact["stable_fraction"] - interval["stable_fraction"],
                bound_reduction=interval["upper_bound"] - exact["upper_bound"])
    if reselect:
        start = time.perf_counter()
        order = deletion_order(Xtr, ytr, seed)
        chosen = list(range(Xtr.shape[1]))
        history = []
        for feature in order:
            candidate = [k for k in chosen if k != feature]
            if not candidate:
                continue
            cert = certificate(Xtr, ytr, candidate, weights, eps, stability="exact")
            accepted = bool(cert["upper_bound"] + 1e-10 <= eta)
            history.append(dict(removed_candidate=int(feature), accepted=accepted,
                                upper_bound=cert["upper_bound"]))
            if accepted:
                chosen = candidate
        final = certificate(Xtr, ytr, chosen, weights, eps, stability="exact")
        assert final["upper_bound"] <= eta + 1e-9
        row["reselection"] = dict(features=chosen, n_features=len(chosen), certificate=final,
                                  seconds=time.perf_counter() - start, history=history,
                                  extra_attributes_removed=len(features) - len(chosen),
                                  same_subset=features == chosen)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--source", default="results/archive/primary-v1")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--eta", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--reselect", action="store_true")
    parser.add_argument("--output", default="results/exact-stability.json")
    args = parser.parse_args()
    import numpy as np
    from certified_frfs.datasets import available_datasets
    names = args.datasets or available_datasets()
    jobs = [(name, args.source, args.seed, args.eps, args.eta, args.reselect) for name in names]
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_dataset, job): job[0] for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps({key: row[key] for key in
                              ("dataset", "stable_fraction_gain", "bound_reduction",
                               "interval_seconds", "exact_seconds")}), flush=True)
    rows.sort(key=lambda row: row["dataset"])
    summary = dict(n_datasets=len(rows),
                   mean_interval_stable_fraction=float(np.mean([r["interval"]["stable_fraction"] for r in rows])),
                   mean_exact_stable_fraction=float(np.mean([r["exact"]["stable_fraction"] for r in rows])),
                   improved_stability_datasets=sum(r["stable_fraction_gain"] > 1e-10 for r in rows),
                   improved_bound_datasets=sum(r["bound_reduction"] > 1e-10 for r in rows),
                   mean_bound_reduction=float(np.mean([r["bound_reduction"] for r in rows])),
                   max_bound_reduction=max(r["bound_reduction"] for r in rows),
                   mean_interval_seconds=float(np.mean([r["interval_seconds"] for r in rows])),
                   mean_exact_seconds=float(np.mean([r["exact_seconds"] for r in rows])))
    if args.reselect:
        changes = [r["reselection"]["extra_attributes_removed"] for r in rows]
        summary.update(reselected_datasets=len(rows), more_reduced_datasets=sum(c > 0 for c in changes),
                       less_reduced_datasets=sum(c < 0 for c in changes),
                       changed_subsets=sum(not r["reselection"]["same_subset"] for r in rows),
                       total_extra_attributes_removed=sum(changes),
                       mean_extra_attributes_removed=float(np.mean(changes)))
    report = dict(exploratory=True, protocol="exact-stability-2026-09-19-v1",
                  comparison=("identical default-selected first-fold subsets and exact same-order reselection"
                              if args.reselect else "identical default-selected first-fold subsets; no reselection"),
                  arguments=vars(args), summary=summary, rows=rows)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
