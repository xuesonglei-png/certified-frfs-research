"""Independent real-data small-table verification, not primary-table gaps.

Fit preprocessing to the seed-17 first training fold. Seed 20260919 chooses
six rows by randomized class round-robin and at most six random columns.
These choices define a NEW decision table, with its own fixed weights.
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


def plain(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def balanced_rows(y, count, rng):
    """Random class order, random within-class order, round-robin allocation.

    With more classes than rows, the first count randomized classes supply
    one row each. Exhausted classes are skipped. This is balanced sampling
    of the constructed instance, not a reweighting of its dependency.
    """
    import numpy as np
    classes = rng.permutation(np.unique(y))
    pools = [list(rng.permutation(np.flatnonzero(y == label))) for label in classes]
    chosen = []
    while len(chosen) < count:
        progress = False
        for pool in pools:
            if pool and len(chosen) < count:
                chosen.append(int(pool.pop()))
                progress = True
        if not progress:
            raise ValueError("Not enough rows to construct requested subtable")
    if len(np.unique(y[chosen])) < 2:
        raise ValueError("Subtable must contain at least two decision classes")
    return np.asarray(chosen, dtype=int)


def run_dataset(job):
    name, seed, fold_seed, eps, time_limit, attack_samples = job
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import MinMaxScaler
    from certified_frfs.core import certificate, gamma
    from certified_frfs.datasets import load_dataset, metadata
    from certified_frfs.verification import exact_loss_milp, sampled_attack

    started = time.perf_counter()
    X, y = load_dataset(name)
    splits = min(3, int(np.unique(y, return_counts=True)[1].min()))
    train, _ = next(StratifiedKFold(splits, shuffle=True, random_state=fold_seed).split(X, y))
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = MinMaxScaler(clip=True)
    Xtr = scaler.fit_transform(imputer.fit_transform(X[train]))
    ytr = y[train]
    rng = np.random.default_rng(seed)
    rows = balanced_rows(ytr, 6, rng)
    columns = (np.sort(rng.choice(Xtr.shape[1], 6, replace=False))
               if Xtr.shape[1] > 6 else np.arange(Xtr.shape[1]))
    table, labels = Xtr[np.ix_(rows, columns)], ytr[rows]
    d = table.shape[1]
    features = list(range(d // 2))
    weights = np.full(d, 1 / np.sqrt(d))
    certificates = {}
    for label, method, stability in [("independent", "independent", "interval"),
                                     ("forest_hybrid", "hybrid", "interval"),
                                     ("exact_stability_hybrid", "hybrid", "exact")]:
        tick = time.perf_counter()
        value = certificate(table, labels, features, weights, eps,
                            method=method, stability=stability)
        certificates[label] = value | {"seconds": time.perf_counter() - tick}
    attack = sampled_attack(table, labels, features, weights, eps,
                            n_samples=attack_samples, seed=seed)
    tick = time.perf_counter()
    exact = exact_loss_milp(table, labels, features, weights, eps,
                           time_limit=time_limit, mip_rel_gap=0.0)
    exact["seconds"] = time.perf_counter() - tick
    for value in certificates.values():
        assert attack["lower_bound"] <= value["upper_bound"] + 1e-8
        assert exact["lower_bound"] <= value["upper_bound"] + 1e-8
        # The amount of certificate conservatism itself has lower/upper
        # bounds until the independent MILP is closed to tolerance.
        value["excess_over_true_loss_lower_bound"] = max(0.0, value["upper_bound"] - exact["upper_bound"])
        value["excess_over_true_loss_upper_bound"] = max(0.0, value["upper_bound"] - exact["lower_bound"])
    assert attack["lower_bound"] <= exact["upper_bound"] + 1e-8
    return plain(dict(dataset=name, construction_seed=seed, first_fold_seed=fold_seed,
                      fold=0, parent_train_size=len(train), parent_d=Xtr.shape[1],
                      source_processed_sha256=metadata(name)["processed_sha256"],
                      parent_training_row_indices=rows, original_row_indices=train[rows],
                      original_column_indices=columns, X=table, y=labels,
                      n=6, d=d, B=features, weights=weights, eps=eps,
                      class_counts={str(label): int(np.sum(labels == label)) for label in np.unique(labels)},
                      nominal_loss=gamma(table, labels, weights=weights) - gamma(table, labels, features, weights),
                      random_attack=attack, certificates=certificates, milp=exact,
                      total_seconds=time.perf_counter() - started))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--fold-seed", type=int, default=17)
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--time-limit", type=float, default=15.0)
    parser.add_argument("--attack-samples", type=int, default=500)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--output", default="results/real-subtable-verification.json")
    args = parser.parse_args()
    import numpy as np
    import scipy
    from certified_frfs.datasets import available_datasets
    names = args.datasets or available_datasets()
    started = time.perf_counter()
    rows = []
    jobs = [(name, args.seed, args.fold_seed, args.eps, args.time_limit, args.attack_samples)
            for name in names]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_dataset, job): job[0] for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(dict(dataset=row["dataset"], lower=row["milp"]["lower_bound"],
                                  upper=row["milp"]["upper_bound"], gap=row["milp"]["gap"],
                                  optimal=row["milp"]["optimal"], seconds=row["milp"]["seconds"])), flush=True)
    rows.sort(key=lambda r: r["dataset"])
    summary = dict(n_instances=len(rows), solver_optimal_instances=sum(r["milp"]["optimal"] for r in rows),
                   max_solver_gap=max(r["milp"]["gap"] for r in rows),
                   mean_nominal_loss=float(np.mean([r["nominal_loss"] for r in rows])),
                   mean_random_attack_lower=float(np.mean([r["random_attack"]["lower_bound"] for r in rows])),
                   mean_milp_lower=float(np.mean([r["milp"]["lower_bound"] for r in rows])),
                   mean_milp_upper=float(np.mean([r["milp"]["upper_bound"] for r in rows])),
                   total_seconds=time.perf_counter() - started)
    for key in ("independent", "forest_hybrid", "exact_stability_hybrid"):
        summary[key] = dict(mean_upper=float(np.mean([r["certificates"][key]["upper_bound"] for r in rows])),
                            mean_excess_lower=float(np.mean([r["certificates"][key]["excess_over_true_loss_lower_bound"] for r in rows])),
                            mean_excess_upper=float(np.mean([r["certificates"][key]["excess_over_true_loss_upper_bound"] for r in rows])))
    report = dict(protocol="real-subtable-verification-2026-09-19-v1", exploratory=True,
                  scope="15 independently constructed small decision tables, not exact verification of primary full training tables",
                  caveats=["Dependency is an object-weighted average, not a per-class or per-object guarantee.",
                           "MILP optimality and certificates use floating-point arithmetic and numerical tolerances.",
                           "Random attack is an attained lower bound, not a robustness certificate."],
                  versions={"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__},
                  arguments=vars(args), summary=summary, rows=rows)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
