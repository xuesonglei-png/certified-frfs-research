"""Verify conditional budgets, common refits, training selections and queries."""
import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from audit_nested_geometry_forest import (ROOT, check_model, score, BA, equivalent, interaction_tree,
    load_dataset, preprocess_fit, CONFIGS, GRIDS, fgs_budget_path, sfrip_budget_path, paired)

FOLDER = ROOT/'results/geometry-forest-matched-count'
ANCHOR = ROOT/'results/nested-geometry-forest'
NATIVE = ROOT/'results/common-fold-recent'


def trees(block, A, labels, counts):
    mapping = {}
    for tree in block['trees']:
        fs = tree['features']; local = interaction_tree(A[:, fs], labels)
        equivalent(local, tree['local_tree'])
        assert tree['edges'] == [sorted([fs[u], fs[v]]) for u, v in local['edges']]
        mapping[tuple(fs)] = tree['edges']; counts['trees'] += 1
    return mapping


def audit(name):
    threadpool_limits(1); counts = Counter()
    artifact = FOLDER/(name+'.json'); saved = json.loads(artifact.read_text())
    anchor = json.loads((ANCHOR/(name+'.json')).read_text()); native = json.loads((NATIVE/(name+'.json')).read_text())
    assert saved['complete'] and anchor['complete']
    for archive in (saved, native):
        for rel, value in archive['fingerprints'].items(): assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == value
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (artifact, Path(__file__), ROOT/'scripts/audit_nested_geometry_forest.py')}
    checkpoint = FOLDER/(name+'-audit.json')
    if checkpoint.exists():
        previous = json.loads(checkpoint.read_text())
        if previous['fingerprints'] == hashes: return previous['result']
    X, y = load_dataset(name); assert len(saved['splits']) == len(anchor['splits'])
    for i, (s, reference) in enumerate(zip(saved['splits'], anchor['splits'])):
        for key in ('reference_indices', 'validation_indices'): assert s[key] == reference[key] == native['splits'][i][key]
        ri, vi = np.array(s['reference_indices']), np.array(s['validation_indices'])
        assert set(ri).isdisjoint(vi)
        chosen = reference['results']['proposed']; anchor_row = reference['trace']['proposed'][chosen['chosen_index']]
        assert s['anchor_chosen_index'] == chosen['chosen_index'] and s['target_count'] == len(chosen['features'])
        assert len(s['inner']) == len(reference['inner'])
        for j, (inner, original) in enumerate(zip(s['inner'], reference['inner'])):
            for key in ('reference_indices', 'validation_indices'): assert inner[key] == original[key]
            a, b = np.array(inner['reference_indices']), np.array(inner['validation_indices'])
            assert set(a).isdisjoint(b) and len(set(a)|set(b)) == len(ri)
            target = len(anchor_row['folds'][j]['features']); assert inner['target_count'] == target
            A, B = preprocess_fit(X[ri[a]], X[ri[b]]); labels = y[ri[a]]
            tree_map = trees(inner, A, labels, counts)
            for fit in inner['models']: check_model(A, labels, fit, counts)
            for method, rows in s['trace'].items():
                for row in rows:
                    ci = row['selector_index']; C, gamma = CONFIGS[row['model_index']]
                    ranking = native['splits'][i]['inner'][j]['paths'][method][ci]['ranking']
                    assert sorted(ranking) == list(range(saved['d']))
                    fs = sorted(ranking[:target]); assert fs == inner['supports'][method][ci]
                    fold = row['folds'][j]; fit = inner['models'][fold['model_id']]
                    assert fit['features'] == fs == fold['features'] and len(fs) == target
                    assert fit['C'] == C and fit['pair_penalty'] == (1. if gamma is None else gamma)
                    assert fit['edges'] == ([] if gamma is None else tree_map[tuple(fs)])
                    np.testing.assert_allclose(fold['clean'], BA(score(B, fit), fit['classes'], y[ri[b]]), atol=1e-12)
                    counts['inner_scores'] += 1; counts['inner_count_matches'] += 1
            counts['inner_splits'] += 1
        A, B = preprocess_fit(X[ri], X[vi]); tree_map = trees(s, A, y[ri], counts)
        assert s['query_seed'] == 144+i
        rng = np.random.default_rng(144+i); queries = [B]+[np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1) for _ in range(3)]
        for method, rows in s['trace'].items():
            assert len(rows) == len(GRIDS[method])*12
            for row in rows:
                np.testing.assert_allclose(row['mean_clean'], np.mean([v['clean'] for v in row['folds']]), atol=1e-12)
                np.testing.assert_allclose(row['mean_size'], np.mean([len(v['features']) for v in row['folds']]), atol=1e-12)
                assert row['fraction_index'] == 0
            selected = max(range(len(rows)), key=lambda j: (round(rows[j]['mean_clean'], 12), -rows[j]['mean_size'], -rows[j]['model_index'], -rows[j]['selector_index']))
            result = s['results'][method]; row = rows[selected]; fit = result['fit']
            assert result['chosen_index'] == selected and result['model_index'] == row['model_index'] and result['selector_index'] == row['selector_index']
            config = GRIDS[method][row['selector_index']]; assert config == result['config']
            path = (fgs_budget_path if method == 'fgs2023' else sfrip_budget_path)(A, y[ri], **config)
            equivalent(path, result['path']); fs = sorted(path['ranking'][:s['target_count']])
            assert result['features'] == fs == fit['features'] and len(fs) == s['target_count']
            C, gamma = CONFIGS[row['model_index']]
            assert fit['C'] == C and fit['pair_penalty'] == (1. if gamma is None else gamma)
            assert fit['edges'] == ([] if gamma is None else tree_map[tuple(fs)])
            check_model(A, y[ri], fit, counts)
            values = [BA(score(Q, fit), fit['classes'], y[vi]) for Q in queries]
            equivalent(result['metrics'], {'clean': values[0], 'noise': float(np.mean(values[1:]))})
            counts['outer_native_path_reruns'] += 1; counts['outer_count_matches'] += 1; counts['outer_predictions'] += 4
        counts['outer_splits'] += 1
        print(name, i, 'matched audit passed', flush=True)
    methods = {m: {metric: float(np.mean([s['results'][m]['metrics'][metric] for s in saved['splits']])) for metric in ('clean', 'noise')} for m in GRIDS}
    methods['proposed'] = {metric: float(np.mean([s['results']['proposed']['metrics'][metric] for s in anchor['splits']])) for metric in ('clean', 'noise')}
    result = {'dataset': name, 'methods': methods, 'counts': dict(counts)}
    checkpoint.write_text(json.dumps({'fingerprints': hashes, 'result': result}, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args()
    summary = None if args.only else json.loads((FOLDER/'complete.json').read_text())
    names = [args.only] if args.only else [r['dataset'] for r in summary['rows']]; rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(audit, name) for name in names]): rows.append(f.result())
    if args.only: print(rows, flush=True); return
    lookup = {r['dataset']: r for r in rows}; rows = [lookup[n] for n in names]
    equivalent(summary['rows'], [{'dataset': r['dataset'], 'methods': r['methods']} for r in rows])
    counts = sum((Counter(r['counts']) for r in rows), Counter())
    assert counts['outer_splits'] == summary['outer_splits'] == 44 and counts['inner_splits'] == 128
    assert counts['outer_count_matches'] == 88
    for metric in ('clean', 'noise'):
        for m in GRIDS: equivalent(summary['contrasts'][metric][m], paired([r['methods']['proposed'][metric]-r['methods'][m][metric] for r in rows]))
    result = {'passed': True, 'counts': dict(counts), 'scope': 'independent_refits_scores_choices_trees_and_exact_count_controls',
        'limits': 'Conditional proposed-selected budgets; inner native rankings fingerprinted cache checked, not all rerun; exposed development.'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); print(result, flush=True)


if __name__ == '__main__': main()
