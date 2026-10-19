"""Reproducible second-cycle counterexamples and small-table exact checks."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from certified_frfs.core import positive_region
from certified_frfs.relative import relative_certificate
from certified_frfs.relative_verification import exact_relative_margin_milp


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_case(X, y, weights, features, eps, clip=True, exact=True):
    X, y, weights, eps = np.array(X, float), np.array(y), np.array(weights), np.array(eps)
    cert = relative_certificate(X, y, features, weights, eps, .9, 'classwise', 'hybrid', clip)
    independent = relative_certificate(X, y, features, weights, eps, .9, 'classwise', 'independent', clip)
    result = {'X': X, 'y': y, 'weights': weights, 'features': features,
              'eps': eps, 'clip': clip, 'retention': .9,
              'independent': independent, 'hybrid': cert, 'exact': []}
    if exact:
        for label in np.unique(y):
            solved = exact_relative_margin_milp(X, y, features, weights, eps, clip,
                         object_weights=(y == label).astype(float), retention=.9, time_limit=10)
            g = next(group for group in cert['groups'] if group['group'] == label)
            assert g['upper_bound'] >= solved['lower_bound']-1e-7
            solved['group'] = label.item()
            result['exact'].append(solved)
    return result


def counterexamples():
    cases = {}
    X = np.array([[0., 0.]] + [[1., 0.]]*18 + [[0., .1]])
    y = np.array([0]*19+[1])
    pc, pb = positive_region(X, y, weights=[1, 1]), positive_region(X, y, [0], [1, 1])
    full_class = [pc[y == c].mean() for c in [0, 1]]
    kept_class = [pb[y == c].mean() for c in [0, 1]]
    cases['rare_class_collapse'] = evaluate_case(X, y, [1, 1], [0], 0.)
    cases['rare_class_collapse'].update(
        global_retention=pb.mean()/pc.mean(),
        macro_retention=sum(kept_class)/sum(full_class),
        class_retention=(np.array(kept_class)/full_class).tolist())
    assert np.isclose(cases['rare_class_collapse']['macro_retention'], .9)

    X = np.array([[.5, 0.], [.5, .02]])
    eps = np.array([[.5, 0.], [0., 0.]])
    cases['interior_violation'] = evaluate_case(X, [0, 1], [1, 1], [0], eps)
    vertex_values = []
    for t in [0., 1.]:
        Z = X.copy(); Z[0, 0] = t
        vertex_values.append(float(.9*positive_region(Z, [0, 1], weights=[1, 1]).mean()
                                   - positive_region(Z, [0, 1], [0], [1, 1]).mean()))
    cases['interior_violation']['all_vertex_margins'] = vertex_values
    assert np.allclose(vertex_values, [-.032, -.032])
    assert all(np.isclose(r['lower_bound'], .018) for r in cases['interior_violation']['exact'])

    X = np.array([[.5, .5], [5., 0.], [0., 0.], [1., 0.]])
    eps = np.zeros_like(X); eps[0, 0] = .1
    cases['star_certifies_extra_deletion'] = evaluate_case(X, [0, 0, 1, 1], [1, .1], [0], eps, False)
    item = cases['star_certifies_extra_deletion']
    assert not item['independent']['certified'] and item['hybrid']['certified']
    assert np.allclose([r['lower_bound'] for r in item['exact']], [-.0475, -.005])

    X = np.array([[0., 0.], [.5, 0.], [.25, 1.], [.75, 1.]])
    y = np.array([0, 0, 1, 1]); weights = np.array([4., 4.])
    pc = positive_region(X, y, weights=weights)
    pb = positive_region(X, y, [0], weights)
    loo_scores = []
    for features in [[0, 1], [0]]:
        dist = cdist(X[:, features]*weights[features], X[:, features]*weights[features], metric='cityblock')
        np.fill_diagonal(dist, np.inf)
        loo_scores.append(float(np.mean(y[dist.argmin(axis=1)] == y)))
    assert np.all(pc == 1) and np.all(pb == 1) and loo_scores == [1., 0.]
    cases['dependency_is_not_classification'] = {
        'X': X, 'y': y, 'weights': weights, 'features': [0],
        'full_positive_region': pc, 'retained_positive_region': pb,
        'full_loo_1nn_accuracy': loo_scores[0], 'retained_loo_1nn_accuracy': loo_scores[1],
        'scope': 'Constructed leave-one-out example, not a real-data predictive experiment',
    }
    return cases


def small_real_tables():
    source = ROOT/'results/real-subtable-verification.json'
    old = json.loads(source.read_text(encoding='utf-8'))
    rows = []
    for entry in old['rows']:
        X, y = np.array(entry['X']), np.array(entry['y'])
        d = X.shape[1]
        features = list(range(d//2))
        item = evaluate_case(X, y, np.ones(d)/np.sqrt(d), features, .01)
        stronger = relative_certificate(X, y, features, np.ones(d)/np.sqrt(d), .01,
                                        .9, 'classwise', 'hybrid', True, 'exact')
        item.update(dataset=entry['dataset'], exact_stability=stronger)
        for solved in item['exact']:
            group = next(g for g in stronger['groups'] if g['group'] == solved['group'])
            assert group['upper_bound'] >= solved['lower_bound']-1e-7
        rows.append(item)
        print(f"{entry['dataset']}: {len(item['exact'])} class verifications, "
              f"all_optimal={all(r['optimal'] for r in item['exact'])}", flush=True)
    return rows, fingerprint(source)


def main():
    started = perf_counter()
    with threadpool_limits(limits=1):
        cases = counterexamples()
        rows, source_hash = small_real_tables()
    comparisons = []
    for row in rows:
        for exact in row['exact']:
            comparison = {'optimal': exact['optimal'], 'gap': exact['gap'],
                          'exact_margin': exact['lower_bound']}
            for key in ['independent', 'hybrid', 'exact_stability']:
                group = next(g for g in row[key]['groups'] if g['group'] == exact['group'])
                comparison[key+'_excess'] = group['upper_bound']-exact['lower_bound']
            comparisons.append(comparison)
    result = {'protocol': 'relative-prototypes-2026-09-20', 'exploratory': True,
              'scope': 'constructed counterexamples and 15 existing induced small decision tables, not whole-primary-table optimality',
              'retention': .9, 'real_table_epsilon': .01,
              'counterexamples': cases, 'real_small_tables': rows,
              'source_real_subtables_sha256': source_hash,
              'script_sha256': fingerprint(Path(__file__)),
              'code_sha256': {str(p.relative_to(ROOT)): fingerprint(p) for p in sorted((ROOT/'src/certified_frfs').glob('*.py'))},
              'summary': {
                  'small_real_tables': len(rows), 'classwise_exact_problems': len(comparisons),
                  'optimal_count': sum(c['optimal'] for c in comparisons),
                  'max_gap': max(c['gap'] for c in comparisons),
                  'mean_exact_margin': float(np.mean([c['exact_margin'] for c in comparisons])),
                  **{f'mean_{key}_excess': float(np.mean([c[key+'_excess'] for c in comparisons]))
                     for key in ['independent', 'hybrid', 'exact_stability']},
              }, 'seconds': perf_counter()-started}
    dest = ROOT/'results/round2/relative-prototypes.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2, default=serial)+'\n', encoding='utf-8')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
