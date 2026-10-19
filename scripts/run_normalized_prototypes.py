"""Frozen, separate normalized-model examples and induced-real-table checks."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.normalized_relative import normalized_relative_certificate
from certified_frfs.normalized_verification import exact_normalized_margin_milp


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    return {str(p.relative_to(ROOT)): digest(p)
            for p in [Path(__file__), *sorted((ROOT/'src/certified_frfs').glob('*.py'))]}


def evaluate(X, y, weights, features, eps):
    X, y, weights = np.asarray(X, float), np.asarray(y), np.asarray(weights, float)
    row = dict(X=X, y=y, weights=weights, features=features, eps=eps, retention=.9)
    for key, method, stability in [('independent', 'independent', 'interval'),
                                    ('hybrid', 'hybrid', 'interval'),
                                    ('exact_stability', 'hybrid', 'exact')]:
        row[key] = normalized_relative_certificate(X, y, features, weights, eps=eps,
                    retention=.9, scope='classwise', method=method, stability=stability)
    row['exact'] = []
    for label in np.unique(y):
        solved = exact_normalized_margin_milp(X, y, features, weights, eps=eps,
                    object_weights=(y == label).astype(float), retention=.9, time_limit=10)
        solved['group'] = label.item()
        for key in ['independent', 'hybrid', 'exact_stability']:
            group = next(g for g in row[key]['groups'] if g['group'] == label)
            assert group['upper_bound'] >= solved['lower_bound']-1e-7
        row['exact'].append(solved)
    return row


def examples():
    cases = {}
    cases['twenty_duplicate_columns'] = evaluate(np.array([[0.]*20, [1.]*20]),
                                               [0, 1], np.ones(20), [0], .01)
    assert all(np.isclose(r['lower_bound'], -.0809, atol=1e-8)
               for r in cases['twenty_duplicate_columns']['exact'])
    X = [[0., 0., 0.], [1., 0., 1.]]
    cases['feasible_sets_not_upward_closed'] = [evaluate(X, [0, 1], np.ones(3), B, 0.)
                                               for B in [[0], [0, 1], [0, 1, 2]]]
    np.testing.assert_allclose([r['exact'][0]['lower_bound']
                               for r in cases['feasible_sets_not_upward_closed']], [-.4, .1, -1/15])
    X = np.array([[.1, .5], [1., 0.], [0., 0.], [.2, 0.]])
    eps = np.zeros_like(X); eps[0, 0] = .02
    case = evaluate(X, [0, 0, 1, 1], [1., .025], [0], eps)
    assert not case['independent']['certified'] and case['hybrid']['certified']
    assert np.isclose(case['exact'][1]['lower_bound'], -.00125/1.025, atol=1e-8)
    case['exactness_scope'] = 'Class 1 shares full/retained stable witnesses; class 0 has switching witnesses'
    cases['normalized_star_certifies_extra_deletion'] = case
    return cases


def main():
    started = perf_counter()
    initial = sources()
    source = ROOT/'results/real-subtable-verification.json'
    old = json.loads(source.read_text(encoding='utf-8'))
    rows = []
    with threadpool_limits(limits=1):
        cases = examples()
        for entry in old['rows']:
            X, y = np.array(entry['X']), np.array(entry['y'])
            d = X.shape[1]
            row = evaluate(X, y, np.ones(d)/np.sqrt(d), list(range(d//2)), .01)
            row['dataset'] = entry['dataset']
            rows.append(row)
            print(f"{row['dataset']}: {len(row['exact'])} classwise MILPs", flush=True)
    comparisons = []
    for row in rows:
        for solved in row['exact']:
            item = {key: solved[key] for key in ['optimal', 'gap', 'lower_bound']}
            for key in ['independent', 'hybrid', 'exact_stability']:
                group = next(g for g in row[key]['groups'] if g['group'] == solved['group'])
                item[key+'_excess'] = group['upper_bound']-solved['lower_bound']
            comparisons.append(item)
    end = sources()
    assert initial == end, 'Source changed during verification'
    result = dict(protocol='normalized-prototypes-2026-09-20', run_status='complete',
        caveat='Different normalized target; induced small tables, not full-training-table exact gaps; numerical MILP, not formal verification',
        counterexamples=cases, real_small_tables=rows, source_real_subtables_sha256=digest(source),
        source_hashes=initial, source_hashes_end=end, source_files_unchanged_during_run=True,
        seconds=perf_counter()-started, completed_utc=datetime.now(timezone.utc).isoformat(),
        summary=dict(small_real_tables=len(rows), classwise_exact_problems=len(comparisons),
            optimal_count=sum(c['optimal'] for c in comparisons), max_gap=max(c['gap'] for c in comparisons),
            mean_exact_margin=float(np.mean([c['lower_bound'] for c in comparisons])),
            **{f'mean_{key}_excess': float(np.mean([c[key+'_excess'] for c in comparisons]))
               for key in ['independent', 'hybrid', 'exact_stability']}))
    target = ROOT/'results/round2/normalized-prototypes.json'
    target.write_text(json.dumps(result, indent=2, default=serial, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
