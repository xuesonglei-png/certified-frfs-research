"""Reproduce the 400-case/40-MILP theory validation with a fixed seed."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--output", default="results/theory_validation.json")
    args = parser.parse_args()
    import numpy as np
    from threadpoolctl import threadpool_limits
    from certified_frfs.core import certificate, interval_bounds, forest_maximum, _distance, _interval_distances
    from certified_frfs.verification import exact_loss_milp, sampled_attack
    threadpool_limits(1)
    rng = np.random.default_rng(args.seed)
    start = time.perf_counter()
    report = dict(seed=args.seed, forest_cases=0, general_cases=0, exact_structure_cases=0,
                  milp_cases=0, max_milp_gap=0.0, max_attack_minus_certificate=-1.0,
                  max_exact_structure_error=0.0)
    for case in range(200):
        n, d = int(rng.integers(2, 9)), int(rng.integers(1, 4))
        lower = rng.uniform(-1, 1, (n, d))
        upper = lower + rng.uniform(0, 2, (n, d))
        edges = [(i, int(rng.integers(i)), float(rng.uniform(.1, 3)))
                 for i in range(1, n) if rng.random() < .85]
        weights = rng.uniform(.1, 1, d)
        expected = 0.0
        for k in range(d):
            best = 0.0
            for bits in itertools.product((0, 1), repeat=n):
                z = np.where(bits, upper[:, k], lower[:, k])
                best = max(best, sum(c * abs(z[i] - z[j]) for i, j, c in edges))
            expected += weights[k] * best
        actual = forest_maximum(lower, upper, edges, weights)
        assert abs(expected - actual) < 1e-9, ("forest", case, expected, actual)
        report["forest_cases"] += 1
    for case in range(100):
        n, d = int(rng.integers(3, 6)), int(rng.integers(2, 4))
        X, y = rng.uniform(0, 1, (n, d)), np.arange(n) % 2
        weights, eps = rng.uniform(0, 1, d), rng.uniform(0, .3, (n, d))
        features = list(range(d - 1))
        certs = [certificate(X, y, features, weights, eps, method=method)
                 for method in ["hybrid", "independent", "lipschitz"]]
        attack = sampled_attack(X, y, features, weights, eps, n_samples=200, seed=case)
        for result in certs:
            discrepancy = attack["lower_bound"] - result["upper_bound"]
            report["max_attack_minus_certificate"] = max(report["max_attack_minus_certificate"], discrepancy)
            assert discrepancy < 1e-9, ("attack", case, discrepancy)
        if case < 20:
            exact = exact_loss_milp(X, y, features, weights, eps, time_limit=5)
            assert exact["lower_bound"] <= certs[0]["upper_bound"] + 1e-7
            assert attack["lower_bound"] <= exact["upper_bound"] + 1e-7
            report["milp_cases"] += 1
            report["max_milp_gap"] = max(report["max_milp_gap"], exact["gap"])
        report["general_cases"] += 1
    attempts = 0
    while report["exact_structure_cases"] < 100:
        attempts += 1
        n = int(rng.integers(3, 8))
        X, y = rng.uniform(0, 1, (n, 3)), np.arange(n) % 2
        weights, features = np.array([.2, .2, .005]), [0, 1]
        eps = np.zeros_like(X)
        eps[:, 2] = rng.uniform(0, .3, n)
        lower, upper = interval_bounds(X, eps)
        distances, cross = _distance(X, features, weights), y[:, None] != y[None, :]
        nearest = np.argmin(np.where(cross, distances, np.inf), axis=1)
        lo, hi = _interval_distances(lower, upper, np.arange(3), weights)
        competitors = np.where(cross, lo, np.inf)
        competitors[np.arange(n), nearest] = np.inf
        if not np.all(hi[np.arange(n), nearest] < competitors.min(axis=1)):
            continue
        assert np.all(hi[np.arange(n), nearest] <= 1)
        result = certificate(X, y, features, weights, eps)
        assert result["stable_count"] == n
        edges = [(i, int(nearest[i]), 1) for i in range(n)]
        expected = forest_maximum(lower[:, [2]], upper[:, [2]], edges, weights[[2]]) / n
        error = abs(expected - result["upper_bound"])
        report["max_exact_structure_error"] = max(report["max_exact_structure_error"], error)
        assert error < 1e-9
        if report["exact_structure_cases"] < 20:
            exact = exact_loss_milp(X, y, features, weights, eps, time_limit=5)
            assert exact["optimal"] and abs(exact["lower_bound"] - expected) < 1e-7
            report["milp_cases"] += 1
            report["max_milp_gap"] = max(report["max_milp_gap"], exact["gap"])
        report["exact_structure_cases"] += 1
    report.update(exact_structure_generation_attempts=attempts,
                  seconds=time.perf_counter() - start, all_assertions_passed=True)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
