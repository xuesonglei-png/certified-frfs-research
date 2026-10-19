"""Independent direct-substitution risk/path/FRS/prediction audit."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from audit_nested_geometry_forest import (ROOT, check_model, score, BA, equivalent, load_dataset,
    preprocess_fit, interaction_tree, weight_certificate, transfer_weights, capped_sparse_projection, paired)
from certified_frfs.nested_teacher_regularized import inner_splits
from certified_frfs.majority_metric import MajorityTerms

FOLDER = ROOT/'results/crossfit-forest-risk-pilot'
ANCHOR = ROOT/'results/nested-geometry-forest'


def audit(name):
    threadpool_limits(1); counts = Counter()
    saved = json.loads((FOLDER/(name+'.json')).read_text()); anchor = json.loads((ANCHOR/(name+'.json')).read_text())
    assert saved['complete']
    for rel, value in saved['fingerprints'].items(): assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == value
    X, y = load_dataset(name); expected = [(i, j, s, inner) for i, s in enumerate(anchor['splits']) for j, inner in enumerate(s['inner'])]
    assert len(expected) == len(saved['records'])
    for record, (i, j, s, inner) in zip(saved['records'], expected):
        assert record['outer_index'] == i and record['inner_index'] == j
        assert record['outer_reference_indices'] == s['reference_indices']
        for key in ('reference_indices', 'validation_indices'): assert record[key] == inner[key]
        ri, a, b = (np.array(record[key]) for key in ('outer_reference_indices', 'reference_indices', 'validation_indices'))
        assert set(a).isdisjoint(b) and len(set(a)|set(b)) == len(ri)
        raw, labels = X[ri[a]], y[ri[a]]; A, B = preprocess_fit(raw, X[ri[b]])
        d = A.shape[1]; original = np.ones(d)/np.sqrt(d); fresh = record['stage']; path = fresh['path']
        protected = MajorityTerms(A, labels, weights=original).bounds(range(d), allowed_enemies=2) > 1e-10
        assert fresh['C'] == 1. and fresh['pair_penalty'] == 1. and fresh['teacher_guards_enforced'] is False
        labels_unique, label_counts = np.unique(labels, return_counts=True)
        singletons = np.flatnonzero(np.isin(labels, labels_unique[label_counts == 1])).tolist()
        assert path['singleton_training_only_indices'] == singletons and path['tie_decimals'] == 12
        regenerated = inner_splits(labels, seed=20261012)
        assert len(path['crossfit_splits']) == len(regenerated)
        fold_data = []
        for fold, (c, e) in zip(path['crossfit_splits'], regenerated):
            assert fold['reference_indices'] == c.tolist() and fold['validation_indices'] == e.tolist()
            assert set(c).isdisjoint(e) and set(singletons).issubset(c) and set(singletons).isdisjoint(e)
            T, Q = preprocess_fit(raw[c], raw[e]); fit = fold['fit']; classes = np.array(fit['classes'])
            assert fit['features'] == list(range(d)) and fit['C'] == fit['pair_penalty'] == 1.
            assert fit['edges'] == interaction_tree(T, labels[c])['edges']
            check_model(T, labels[c], fit, counts)
            weights = np.array([1/(len(classes)*sum(labels[c] == z)) for z in labels[c]])
            center = weights @ T; np.testing.assert_allclose(fold['center'], center, atol=1e-12)
            assert fold['query_classes'] == np.unique(labels[e]).tolist()
            own = np.argmax(labels[e, None] == classes[None], axis=1)
            balance = np.array([1/(len(np.unique(labels[e]))*sum(labels[e] == z)) for z in labels[e]])
            fold_data.append((Q, fit, center, own, balance))
            counts['fold_teachers'] += 1; counts['fold_teacher_caps'] += not fit['optimizer_converged']
        losses = {}; geometries = {}
        def loss(fs):
            key = tuple(fs)
            if key not in losses:
                values = []
                for Q, fit, center, own, balance in fold_data:
                    substituted = Q.copy(); omitted = [z for z in range(d) if z not in fs]
                    substituted[:, omitted] = center[omitted]
                    scores = score(substituted, fit)
                    values.append(float(balance @ (logsumexp(scores, axis=1)-scores[np.arange(len(own)), own])))
                losses[key] = float(np.mean(values)); counts['direct_substitution_risk_probes'] += 1
            return losses[key]
        def candidate(r, fs, value):
            assert r['features'] == fs
            np.testing.assert_allclose(r['crossfit_balanced_logloss'], value, atol=1e-11, rtol=1e-9)
            key = tuple(fs)
            if key not in geometries:
                weights = original if len(fs) == d else capped_sparse_projection(transfer_weights(A, fs, original)/np.sqrt(d), len(fs), 2/d)*np.sqrt(d)
                embedded = np.zeros(d); embedded[fs] = weights
                cert = weight_certificate(A, labels, fs, original, weights, protected=protected)
                geometries[key] = (embedded, cert); counts['geometry'] += 1
            embedded, cert = geometries[key]
            np.testing.assert_allclose(r['weights'], embedded, atol=1e-12); equivalent(r['certificate'], cert)
            return cert['feasible']
        active = list(range(d)); ti = ai = 0; accepted = False
        assert candidate(path['trace'][0], active, loss(active))
        for ranking in path['rankings']:
            assert ranking['features'] == active and len(active) > int(np.ceil(d/2))
            ordered = sorted([(loss([z for z in active if z != removed]), removed) for removed in active], key=lambda r: (round(r[0], 12), r[1]))
            assert [r['removed'] for r in ranking['choices']] == [removed for _, removed in ordered]
            np.testing.assert_allclose([r['loss'] for r in ranking['choices']], [value for value, _ in ordered], atol=1e-11, rtol=1e-9)
            accepted = False
            for value, removed in ordered:
                attempt = path['attempts'][ai]; ai += 1; fs = [z for z in active if z != removed]
                assert attempt['removed'] == removed
                feasible = candidate(attempt, fs, value); assert attempt['accepted'] == feasible
                counts['attempts'] += 1; counts['geometry_rejections'] += not feasible
                if feasible:
                    ti += 1; assert candidate(path['trace'][ti], fs, value); active = fs; accepted = True; break
            if not accepted: assert ranking is path['rankings'][-1]
        assert ai == len(path['attempts']) and ti+1 == len(path['trace'])
        assert len(active) == int(np.ceil(d/2)) or not accepted
        for fraction in (.5, .65, .8, 1.):
            chosen = next(r for r in reversed(path['trace']) if len(r['features']) >= int(np.ceil(fraction*d)))
            policy = fresh['policies'][str(fraction)]; equivalent(policy['candidate'], chosen)
            fit = policy['fit']; assert fit['features'] == chosen['features'] and fit['C'] == fit['pair_penalty'] == 1.
            assert fit['edges'] == [sorted([chosen['features'][u], chosen['features'][v]]) for u, v in interaction_tree(A[:, chosen['features']], labels)['edges']]
            check_model(A, labels, fit, counts)
        old_stage = inner['stages'][7]
        assert record['fits']['old'] == inner['models'][old_stage['policies']['0.8']['model_id']]
        assert record['fits']['full'] == inner['models'][old_stage['policies']['1.0']['model_id']]
        assert record['fits']['crossfit'] == fresh['policies']['0.8']['fit']
        old_count = len(record['fits']['old']['features'])
        matched = next(r for r in reversed(path['trace']) if len(r['features']) >= old_count)
        assert record['fits']['matched']['features'] == matched['features']
        assert record['matched_count_exact'] == (len(matched['features']) == old_count)
        counts['matched_count_mismatches'] += not record['matched_count_exact']
        rng = np.random.default_rng(144+i); assert record['query_seed'] == 144+i
        queries = [B]+[np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1) for _ in range(3)]
        for m, fit in record['fits'].items():
            check_model(A, labels, fit, Counter())
            values = [BA(score(Q, fit), fit['classes'], y[ri[b]]) for Q in queries]
            equivalent(record['metrics'][m], {'clean': values[0], 'noise': float(np.mean(values[1:])), 'removed': 1-len(fit['features'])/d})
            counts['query_predictions'] += 4; counts['final_fit_caps'] += not fit['optimizer_converged']
        counts['records'] += 1
        print(name, i, j, 'risk audit passed', flush=True)
    return {'dataset': name, 'counts': dict(counts), 'methods': {m: {metric: float(np.mean([r['metrics'][m][metric] for r in saved['records']])) for metric in ('clean', 'noise', 'removed')} for m in ('old', 'crossfit', 'matched', 'full')}}


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
    for key in ('records', 'matched_count_mismatches', 'final_fit_caps', 'fold_teacher_caps'): assert counts[key] == summary[key]
    assert counts['records'] == 128
    for m in ('old', 'crossfit', 'matched', 'full'):
        for metric in ('clean', 'noise', 'removed'): equivalent(summary['means'][m][metric], float(np.mean([r['methods'][m][metric] for r in rows])))
    for metric in ('clean', 'noise'):
        for m in ('crossfit', 'matched', 'full'): equivalent(summary['contrasts'][metric][m], paired([r['methods'][m][metric]-r['methods']['old'][metric] for r in rows]))
    gate = summary['contrasts']['clean']['crossfit']['mean_pp'] >= .5 and summary['means']['crossfit']['removed'] >= summary['means']['old']['removed'] and counts['final_fit_caps'] == counts['fold_teacher_caps'] == 0
    assert summary['gate_passed'] == gate
    result = {'passed': True, 'counts': dict(counts), 'auditor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'direct_raw_substitution_fold_refits_centers_FRScertificates_rankings_final_fits_and_queries',
        'limits': 'Fixed_config_all128_exposed_development_no_nested_superiority_or_population_risk_bound'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); print(result, flush=True)


if __name__ == '__main__': main()
