"""Independent design/refit, raw substitution, certificates and selection audit.

Inner native paths retain fingerprint-verified cache provenance. Selected outer
native paths are recomputed. This is development validation, not confirmation.
"""
import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from review_box_iterations import ROOT, paired
from audit_forest_compact_transfer import equivalent
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.fuzzy_interaction_tree import interaction_tree
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.redistributed_loss import transfer_weights
from certified_frfs.capped_sparse import capped_sparse_projection
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.recent_budget import fgs_budget_path, sfrip_budget_path
from compare_trajectory_recent import GRIDS

FOLDER = ROOT/'results/nested-geometry-forest'
SOURCE = ROOT/'results/nested-crossfit-classification-rounded'
NATIVE = ROOT/'results/common-fold-recent'
CONFIGS = [(c, g) for c in (.1, 1., 10.) for g in (None, 10., 3., 1.)]
METHODS = ('proposed', 'full', 'fgs2023', 'sfrip2025')
FRACTIONS = (.5, .65, .8, 1.)
Q = np.array([[1/np.sqrt(2), 1/np.sqrt(6)], [-1/np.sqrt(2), 1/np.sqrt(6)], [0., -2/np.sqrt(6)]])


def membership(X):
    return np.stack([np.maximum(1-2*X, 0), 1-np.abs(2*X-1), np.maximum(2*X-1, 0)], axis=2)


def score(X, fit):
    Z = membership(X)
    result = np.einsum('ndk,qdk->nq', Z, np.asarray(fit['coef'])) + np.asarray(fit['intercept'])
    for e, (u, v) in enumerate(fit['edges']):
        result += np.einsum('nk,qkl,nl->nq', Z[:, u], np.asarray(fit['pair_coef'])[:, e], Z[:, v])
    return result


def BA(scores, classes, y):
    predicted = np.asarray(classes)[scores.argmax(axis=1)]
    return float(np.mean([np.mean(predicted[y == c] == c) for c in np.unique(y)]))


def check_model(A, y, fit, counts):
    fs, edges = fit['features'], fit['edges']; q, d = len(np.unique(y)), A.shape[1]
    assert fit['classes'] == np.unique(y).tolist() and fit['class_balanced']
    assert fit['teacher_guards_enforced'] is False
    W = np.asarray(fit['coef']); V = np.asarray(fit['pair_coef']).reshape(q, len(edges), 3, 3)
    np.testing.assert_allclose(W[:, [a for a in range(d) if a not in fs]], 0, atol=1e-12)
    np.testing.assert_allclose(W.sum(axis=2), 0, atol=1e-10)
    np.testing.assert_allclose(V.sum(axis=2), 0, atol=1e-10)
    np.testing.assert_allclose(V.sum(axis=3), 0, atol=1e-10)
    if not fs:
        np.testing.assert_allclose(score(A, fit), 0, atol=1e-12)
        assert not edges and fit['optimizer_iterations'] == 0
    else:
        Z = membership(A) @ Q
        blocks = [Z[:, fs].reshape(len(A), -1)]
        blocks += [(Z[:, u, :, None]*Z[:, v, None, :]).reshape(len(A), 4)/np.sqrt(fit['pair_penalty']) for u, v in edges]
        design = np.column_stack(blocks)
        model = LogisticRegression(C=fit['C'], class_weight='balanced', max_iter=4000, tol=1e-8, random_state=0).fit(design, y)
        coef, bias = model.coef_, model.intercept_
        if q == 2:
            coef = np.vstack([np.zeros_like(coef[0]), coef[0]]); bias = np.r_[0., bias[0]]
        decoded = [(W[:, fs] @ Q).reshape(q, -1)]
        decoded += [(np.einsum('ki,qkl,lj->qij', Q, V[:, e], Q)*np.sqrt(fit['pair_penalty'])).reshape(q, 4) for e in range(len(edges))]
        np.testing.assert_allclose(np.column_stack(decoded), coef-coef[:1], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(fit['intercept'], bias-bias[:1], atol=1e-8, rtol=1e-8)
        expected = design @ coef.T + bias; actual = score(A, fit)
        np.testing.assert_allclose(actual-actual[:, :1], expected-expected[:, :1], atol=1e-8, rtol=1e-8)
        assert fit['optimizer_iterations'] == int(model.n_iter_.max())
        assert fit['optimizer_converged'] == (model.n_iter_.max() < 4000)
        counts['independent_classifier_refits'] += 1
    counts['models'] += 1; counts['optimizer_caps'] += not fit['optimizer_converged']


def audit_block(block, raw, labels, counts):
    A, _ = preprocess_fit(raw, raw); d = A.shape[1]
    original = np.ones(d)/np.sqrt(d)
    protected = MajorityTerms(A, labels, weights=original).bounds(range(d), allowed_enemies=2) > 1e-10
    for fit in block['models']: check_model(A, labels, fit, counts)
    for tree in block['trees']:
        fs = tree['features']; local = interaction_tree(A[:, fs], labels)
        equivalent(tree['local_tree'], local)
        assert tree['edges'] == [sorted([fs[u], fs[v]]) for u, v in local['edges']]
        counts['trees'] += 1
    tree_map = {tuple(t['features']): t['edges'] for t in block['trees']}
    for fit in block['models']:
        if fit['edges']: assert fit['edges'] == tree_map[tuple(fit['features'])]
    for geometry in block['geometry']:
        fs = geometry['features']
        weights = original if len(fs) == d else capped_sparse_projection(transfer_weights(A, fs, original)/np.sqrt(d), len(fs), 2/d)*np.sqrt(d)
        embedded = np.zeros(d); embedded[fs] = weights
        np.testing.assert_allclose(geometry['weights'], embedded, atol=1e-12)
        equivalent(geometry['certificate'], weight_certificate(A, labels, fs, original, weights, protected=protected))
        counts['geometry'] += 1
    stages = block['stages']; stages = list(stages.values()) if isinstance(stages, dict) else stages
    for stage in stages:
        assert stage['teacher_guards_enforced'] is False
        assert stage['original_majority_protected_counts'] == {str(c): int(np.sum(protected & (labels == c))) for c in np.unique(labels)}
        path = stage['path']; assert path['tie_decimals'] == 12
        full_fit = block['models'][stage['policies']['1.0']['model_id']]
        C, gamma = stage['C'], stage['pair_penalty']
        assert full_fit['features'] == list(range(d)) and full_fit['C'] == C
        assert full_fit['pair_penalty'] == (1. if gamma is None else gamma)
        assert bool(full_fit['edges']) == (gamma is not None and d > 1)
        weights = np.array([1/(len(np.unique(labels))*sum(labels == c)) for c in labels])
        center = weights @ A; np.testing.assert_allclose(path['center'], center, atol=1e-12)
        target = score(A, full_fit); target -= target.mean(axis=1, keepdims=True)
        def loss(fs):
            replaced = A.copy(); omitted = [a for a in range(d) if a not in fs]
            replaced[:, omitted] = center[omitted]
            projected = score(replaced, full_fit); projected -= projected.mean(axis=1, keepdims=True)
            return float(weights @ np.mean((projected-target)**2, axis=1))
        def candidate(record, fs, value):
            geometry = block['geometry'][record['geometry_id']]
            assert geometry['features'] == fs
            np.testing.assert_allclose(record['projection_loss'], value, atol=1e-11, rtol=1e-9)
            return geometry['certificate']['feasible']
        active = list(range(d)); ti = ai = 0; accepted = False
        assert candidate(path['trace'][0], active, 0.)
        for ranking in path['rankings']:
            assert ranking['features'] == active and len(active) > int(np.ceil(d/2))
            ordered = sorted([(loss([a for a in active if a != j]), j) for j in active], key=lambda v: (round(v[0], 12), v[1]))
            assert [v['removed'] for v in ranking['choices']] == [j for _, j in ordered]
            np.testing.assert_allclose([v['loss'] for v in ranking['choices']], [v for v, _ in ordered], atol=1e-11, rtol=1e-9)
            accepted = False
            for value, removed in ordered:
                attempt = path['attempts'][ai]; ai += 1; fs = [a for a in active if a != removed]
                assert attempt['removed'] == removed
                feasible = candidate(attempt, fs, value); assert feasible == attempt['accepted']
                counts['attempts'] += 1; counts['geometry_rejections'] += not feasible
                if feasible:
                    ti += 1; assert candidate(path['trace'][ti], fs, value)
                    active = fs; accepted = True; break
            if not accepted: assert ranking is path['rankings'][-1]
        assert ai == len(path['attempts']) and ti+1 == len(path['trace'])
        assert len(active) == int(np.ceil(d/2)) or not accepted
        for fraction in FRACTIONS:
            chosen = next(v for v in reversed(path['trace']) if len(block['geometry'][v['geometry_id']]['features']) >= int(np.ceil(fraction*d)))
            policy = stage['policies'][str(fraction)]
            assert policy['geometry_id'] == chosen['geometry_id']
            fit = block['models'][policy['model_id']]
            assert fit['features'] == block['geometry'][chosen['geometry_id']]['features'] and fit['C'] == C
            assert fit['pair_penalty'] == (1. if gamma is None else gamma)
            assert fit['edges'] == ([] if gamma is None else tree_map[tuple(fit['features'])])
        counts['stages'] += 1


def audit_dataset(name):
    threadpool_limits(1); counts = Counter()
    auditor_hash = hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()
    artifact_hash = hashlib.sha256((FOLDER/(name+'.json')).read_bytes()).hexdigest()
    checkpoint = FOLDER/(name+'-audit.json')
    saved = json.loads((FOLDER/(name+'.json')).read_text())
    source = json.loads((SOURCE/(name+'.json')).read_text()); native = json.loads((NATIVE/(name+'.json')).read_text())
    assert saved['complete'] and saved['teacher_guards_enforced'] is False
    for archive in (saved, native):
        for rel, digest in archive['fingerprints'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
    if checkpoint.exists():
        previous = json.loads(checkpoint.read_text())
        if previous['auditor_sha256'] == auditor_hash and previous['artifact_sha256'] == artifact_hash:
            return previous['result']
    X, y = load_dataset(name); assert len(saved['splits']) == len(source['splits'])
    for i, (s, original) in enumerate(zip(saved['splits'], source['splits'])):
        for key in ('reference_indices', 'validation_indices'): assert s[key] == original[key] == native['splits'][i][key]
        ri, vi = np.array(s['reference_indices']), np.array(s['validation_indices'])
        # Archived outer protocol may cap/subsample each reference. Preserve
        # its exact partition rather than assuming the union exhausts X.
        assert set(ri).isdisjoint(vi) and len(set(ri)) == len(ri) and len(set(vi)) == len(vi)
        assert np.all((ri >= 0) & (ri < len(X))) and np.all((vi >= 0) & (vi < len(X)))
        assert len(s['inner']) == len(original['inner'])
        for j, (inner, old) in enumerate(zip(s['inner'], original['inner'])):
            for key in ('reference_indices', 'validation_indices'): assert inner[key] == old[key] == native['splits'][i]['inner'][j][key]
            a, b = np.array(inner['reference_indices']), np.array(inner['validation_indices'])
            assert set(a).isdisjoint(b) and len(set(a)|set(b)) == len(ri)
            assert inner['native_path_source'] == {'outer_index': i, 'inner_index': j}
            audit_block(inner, X[ri[a]], y[ri[a]], counts)
            _, B = preprocess_fit(X[ri[a]], X[ri[b]])
            for method, rows in s['trace'].items():
                for row in rows:
                    fold = row['folds'][j]; fit = inner['models'][fold['model_id']]
                    C, gamma = CONFIGS[row['model_index']]
                    assert fit['C'] == C and fit['pair_penalty'] == (1. if gamma is None else gamma)
                    assert fit['features'] == fold['features']
                    if method == 'proposed':
                        assert row['fraction'] == FRACTIONS[row['fraction_index']]
                        assert fold['model_id'] == inner['stages'][row['model_index']]['policies'][str(row['fraction'])]['model_id']
                    elif method == 'full': assert fit['features'] == list(range(saved['d']))
                    else:
                        assert fit['features'] == sorted(native['splits'][i]['inner'][j]['paths'][method][row['selector_index']]['native']['features'])
                    np.testing.assert_allclose(fold['clean'], BA(score(B, fit), fit['classes'], y[ri[b]]), atol=1e-12)
                    counts['inner_scores'] += 1
            counts['inner_splits'] += 1
        audit_block(s, X[ri], y[ri], counts)
        A, B = preprocess_fit(X[ri], X[vi]); rng = np.random.default_rng(144+i)
        assert s['query_seed'] == 144+i
        queries = [B]+[np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1) for _ in range(3)]
        for method, rows in s['trace'].items():
            assert len(rows) == {'proposed': 48, 'full': 12, 'fgs2023': 108, 'sfrip2025': 72}[method]
            for row in rows:
                np.testing.assert_allclose(row['mean_clean'], np.mean([f['clean'] for f in row['folds']]), atol=1e-12)
                np.testing.assert_allclose(row['mean_size'], np.mean([len(f['features']) for f in row['folds']]), atol=1e-12)
            chosen = max(range(len(rows)), key=lambda j: (round(rows[j]['mean_clean'], 12), -rows[j]['mean_size'], -rows[j]['model_index'], -rows[j]['selector_index'], -rows[j]['fraction_index']))
            r = s['results'][method]; row = rows[chosen]; fit = s['models'][r['model_id']]
            assert r['chosen_index'] == chosen and r['model_index'] == row['model_index'] and r['selector_index'] == row['selector_index']
            C, gamma = CONFIGS[row['model_index']]
            assert fit['features'] == r['features'] and fit['C'] == C and fit['pair_penalty'] == (1. if gamma is None else gamma)
            if method == 'proposed':
                policy = s['stages'][method]['policies'][str(row['fraction'])]
                assert r['model_id'] == policy['model_id'] and r['geometry_id'] == policy['geometry_id']
                assert s['geometry'][r['geometry_id']]['certificate']['feasible']
            elif method == 'full': assert fit['features'] == list(range(saved['d']))
            else:
                config = GRIDS[method][row['selector_index']]
                assert s['native_paths'][method]['config'] == config
                function = fgs_budget_path if method == 'fgs2023' else sfrip_budget_path
                regenerated = function(A, y[ri], **config)
                equivalent(regenerated, s['native_paths'][method]['path'])
                assert r['features'] == sorted(regenerated['native']['features'])
                counts['outer_native_path_reruns'] += 1
            values = [BA(score(Qi, fit), fit['classes'], y[vi]) for Qi in queries]
            equivalent(r['metrics'], {'clean': values[0], 'noise': float(np.mean(values[1:]))})
            counts['outer_predictions'] += 4
        counts['outer_splits'] += 1
        print(name, i, 'audit passed', flush=True)
    result = {'dataset': name, 'counts': dict(counts), 'methods': {m: {
        **{metric: float(np.mean([s['results'][m]['metrics'][metric] for s in saved['splits']])) for metric in ('clean', 'noise')},
        'removed': float(np.mean([1-len(s['results'][m]['features'])/saved['d'] for s in saved['splits']]))} for m in METHODS}}
    checkpoint.write_text(json.dumps({'auditor_sha256': auditor_hash, 'artifact_sha256': artifact_hash, 'result': result}, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args()
    names = [args.only] if args.only else [r['dataset'] for r in json.loads((FOLDER/'complete.json').read_text())['rows']]
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(audit_dataset, name) for name in names]): rows.append(f.result())
    if args.only:
        print(rows, flush=True); return
    lookup = {r['dataset']: r for r in rows}; rows = [lookup[n] for n in names]
    counts = sum((Counter(r['counts']) for r in rows), Counter())
    summary = json.loads((FOLDER/'complete.json').read_text())
    equivalent(summary['rows'], [{'dataset': r['dataset'], 'methods': r['methods']} for r in rows])
    for key in ('models', 'optimizer_caps', 'outer_splits', 'inner_splits'): assert summary['counts'][key] == counts[key]
    assert counts['outer_splits'] == 44 and counts['inner_splits'] == 128
    for metric in ('clean', 'noise', 'removed'):
        for method in METHODS:
            equivalent(summary['means'][method][metric], float(np.mean([r['methods'][method][metric] for r in rows])))
    for metric in ('clean', 'noise'):
        for method in METHODS[1:]:
            equivalent(summary[metric+'_contrasts'][method], paired([r['methods']['proposed'][metric]-r['methods'][method][metric] for r in rows]))
    pvalues = {m: summary['clean_contrasts'][m]['sign_flip_two_sided_p'] for m in METHODS[1:]}
    previous = 0.; adjusted = {}
    for j, method in enumerate(sorted(pvalues, key=pvalues.get)):
        previous = max(previous, min(1., (3-j)*pvalues[method])); adjusted[method] = previous
    equivalent(summary['descriptive_holm_p'], adjusted)
    gate = all(summary['clean_contrasts'][m]['mean_pp'] >= .5 for m in ('fgs2023', 'sfrip2025')) and summary['clean_contrasts']['full']['mean_pp'] >= 0 and summary['means']['proposed']['removed'] >= .1 and counts['optimizer_caps']/counts['models'] <= .1
    assert summary['gate_passed'] == gate
    result = {'passed': True, 'counts': dict(counts), 'auditor_sha256': hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),
        'scope': 'independent_design_refits_direct_substitution_geometry_selection_predictions_selected_outer_native_reruns',
        'limits': 'Inner native selector paths checked against fingerprinted caches, not all rerun; exposed development data; no teacher-margin guarantee.'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result, flush=True)


if __name__ == '__main__': main()
