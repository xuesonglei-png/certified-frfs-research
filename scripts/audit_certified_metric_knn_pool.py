"""Independent coordinate distance, votes, query boxes and nested selection."""
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from run_certified_metric_knn_pool import ROOT, FOLDER, PARENT, METHODS, COUNTS, digest
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.redistributed_loss import transfer_weights
from certified_frfs.capped_sparse import capped_sparse_projection
from audit_forest_compact_transfer import equivalent
from review_box_iterations import paired
from run_nested_joint_path import holm


def independent_evaluate(A, y, B, truth, fs, weights):
    classes = np.unique(y)
    if fs:
        # Accumulate explicit transformed coordinates in feature order; no
        # cdist, production vote routine or fitted classifier is reused.
        distance = np.zeros((len(B), len(A)))
        for a, w in zip(fs, weights): distance += abs(B[:, a, None]*w-A[None, :, a]*w)
        nn = np.argsort(distance, axis=1, kind='stable')[:, :5]
        votes = np.column_stack([np.sum(y[nn] == c, axis=1) for c in classes])
        probabilities = (votes+1)/(5+len(classes)); pred = classes[votes.argmax(axis=1)]
    else:
        probabilities = np.tile([(sum(y == c)+1)/(len(y)+len(classes)) for c in classes], (len(B), 1))
        pred = classes[probabilities.argmax(axis=1)]; nn = np.zeros((len(B), 0), int)
    def balanced(value): return float(np.mean([np.mean(value[truth == c]) for c in np.unique(truth)]))
    own = np.searchsorted(classes, truth)
    metrics = {'clean_BA': balanced(pred == truth), 'balanced_logloss': balanced(-np.log(probabilities[np.arange(len(B)), own])),
               'macro_F1': float(f1_score(truth, pred, average='macro', zero_division=0))}
    return {'predictions': pred.tolist(), 'probabilities': probabilities.tolist(), 'neighbor_indices': nn.tolist(), 'metrics': metrics}


def query_certificate(A, y, B, targets, fs, weights):
    if not fs: return {'certified': [False]*len(B), 'scope': 'no_neighborhood_certificate_for_empty_prior'}
    alo, ahi = np.maximum(0, A-.01), np.minimum(1, A+.01)
    blo, bhi = np.maximum(0, B-.01), np.minimum(1, B+.01)
    distance = np.zeros((len(B), len(A)))
    for a, w in zip(fs, weights): distance += abs(B[:, a, None]*w-A[None, :, a]*w)
    lower = np.full(len(B), -np.inf)
    for i, target in enumerate(targets):
        friends, enemies = np.flatnonzero(y == target), np.flatnonzero(y != target)
        if len(friends) < 3: continue
        if len(enemies) <= 2: lower[i] = np.inf; continue
        friends = friends[np.lexsort((friends, distance[i, friends]))][:3]
        terms = np.zeros((len(enemies), 3))
        for a, w in zip(fs, weights):
            endpoints = []
            for t in (blo[i, a], bhi[i, a]):
                near = np.maximum(0, np.maximum(alo[enemies, a]-t, t-ahi[enemies, a]))
                far = np.maximum(abs(t-alo[friends, a]), abs(t-ahi[friends, a]))
                endpoints.append(near[:, None]-far[None, :])
            terms += w*np.minimum(*endpoints)
        bound = np.partition(terms.min(axis=1), 2)[2]
        lower[i] = bound-1e-10*(1+abs(bound))
    flags = lower > 1e-10
    return {'certified': flags.tolist(), 'lower_bounds': [float(v) if np.isfinite(v) else None for v in lower],
            'coverage': float(flags.mean()), 'k': 5, 'q': 3, 'allowed_enemies': 2,
            'scope': 'fixed_nominal_query_label_shared_reference_and_query_boxes_not_correctness',
            'rounding': 'ordinary_float_with_conservative_tolerance_not_directed_rounding'}


def policies(A, block, count=None):
    d = A.shape[1]; full = list(range(d)); uniform = np.ones(d)/np.sqrt(d)
    lookup = {tuple(r['features']): r for r in block['pool']}
    out = {m: [] for m in METHODS}
    for recipe in block['recipes']:
        r = lookup[tuple(recipe['features'])]; fs = recipe['features']
        if r['accepted']:
            out['proposed'].append({'features': fs, 'weights': np.array(r['selected_metric']['weights'])[fs].tolist(), 'origin': 'admitted_metric'})
        else:
            out['proposed'].append({'features': full, 'weights': uniform.tolist(), 'origin': 'explicit_full_fallback'})
        out['unguarded_pool'].append({'features': fs, 'weights': np.array(r['ordinary']['weights'])[fs].tolist(), 'origin': 'ordinary_uncensored_metric'})
    out['full'] = [{'features': full, 'weights': uniform.tolist(), 'origin': 'uniform_full'}]
    for prefix, method in (('fgs', 'fgs2023'), ('sfrip', 'sfrip2025')):
        for path in block['paths'][method]:
            fs = sorted(path['native']['features'])
            out[prefix+'_native'].append({'features': fs, 'weights': np.ones(len(fs)).tolist(), 'origin': 'uniform_native'})
            if count is not None:
                fs = sorted(path['ranking'][:count])
                w = uniform if len(fs) == d else capped_sparse_projection(transfer_weights(A, fs, uniform)/np.sqrt(d), len(fs), 2/d)*np.sqrt(d)
                out[prefix+'_matched'].append({'features': fs, 'weights': w.tolist(), 'origin': 'ordinary_transferred_capped_metric'})
                out[prefix+'_uniform_matched'].append({'features': fs, 'weights': np.ones(len(fs)).tolist(), 'origin': 'uniform_count_matched'})
    return out


def audit(name):
    threadpool_limits(1)
    artifact = FOLDER/(name+'.json'); saved = json.loads(artifact.read_text())
    assert saved['complete'] and not saved['query_labels_used_for_selection']
    for rel, sha in saved['fingerprints'].items(): assert digest(ROOT/rel) == sha, rel
    parent = json.loads((PARENT/(name+'.json')).read_text())
    parent_audit = json.loads((PARENT/'complete-audit.json').read_text())
    assert parent_audit['passed'] and parent_audit['artifact_hashes'][name] == digest(PARENT/(name+'.json'))
    assert saved['reference_indices'] == parent['reference_indices'] and saved['validation_indices'] == parent['validation_indices']
    X, y = load_dataset(name); ri, vi = np.array(saved['reference_indices']), np.array(saved['validation_indices'])
    raw, labels = X[ri], y[ri]; counts = Counter(datasets=1)
    combined, selected = saved['selection']['options'], saved['selection']['selected']
    for m in METHODS:
        assert len(combined[m]) == COUNTS[m]
        for i, row in enumerate(combined[m]):
            assert row['selector_index'] == i
            folds = []
            for block in saved['inner']:
                op = block['options'][m][i]
                folds.append({**op['evaluation']['metrics'], 'actual_count': len(op['metric']['features'])})
            equivalent(folds, row['folds'])
            for k in ('clean_BA', 'balanced_logloss', 'actual_count'):
                equivalent(row['mean_'+k], float(np.mean([f[k] for f in folds])))
        chosen = min(range(len(combined[m])), key=lambda i: (-round(combined[m][i]['mean_clean_BA'], 12), round(combined[m][i]['mean_balanced_logloss'], 12), combined[m][i]['mean_actual_count'], i))
        assert chosen == selected[m]
    for block, source in zip(saved['inner'], parent['inner']):
        assert block['reference_indices'] == source['reference_indices'] and block['validation_indices'] == source['validation_indices']
        ca, cb = np.array(block['reference_indices']), np.array(block['validation_indices'])
        assert not set(ca)&set(cb)
        A, B = preprocess_fit(raw[ca], raw[cb])
        initial = policies(A, source); count = len(initial['proposed'][selected['proposed']]['features'])
        assert count == block['matched_proposed_count']; options = policies(A, source, count=count)
        for m in METHODS:
            assert len(options[m]) == len(block['options'][m])
            for metric, record in zip(options[m], block['options'][m]):
                equivalent(metric, record['metric'])
                report = independent_evaluate(A, labels[ca], B, labels[cb], metric['features'], metric['weights'])
                equivalent(report, record['evaluation']); counts['inner_predictions'] += 1
    A, B = preprocess_fit(raw, X[vi]); initial = policies(A, parent['outer'])
    count = len(initial['proposed'][selected['proposed']]['features']); assert count == saved['outer']['matched_proposed_count']
    options = policies(A, parent['outer'], count=count); metrics, sizes = {}, {}
    for m in METHODS:
        record = saved['outer']['results'][m]; metric = options[m][selected[m]]
        assert record['selector_index'] == selected[m]; equivalent(metric, record['metric'])
        fs, w = metric['features'], metric['weights']
        report = independent_evaluate(A, labels, B, y[vi], fs, w)
        cert = query_certificate(A, labels, B, np.array(report['predictions']), fs, w)
        equivalent(cert, record['query_certificate'])
        correct = (np.array(report['predictions']) == y[vi]) & np.array(cert['certified'])
        report['metrics']['query_certified_BA'] = float(np.mean([np.mean(correct[y[vi] == c]) for c in np.unique(y[vi])]))
        noise = []
        assert len(record['perturbation_evaluations']) == 16
        for draw, stored in enumerate(record['perturbation_evaluations']):
            rng = np.random.default_rng(20261060+draw)
            noisy_A = np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1)
            noisy_B = np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1)
            fresh = independent_evaluate(noisy_A, labels, noisy_B, y[vi], fs, w)
            equivalent(fresh, stored); noise.append(fresh['metrics']['clean_BA']); counts['shared_perturbation_predictions'] += 1
            changed = np.array(fresh['predictions']) != report['predictions']
            assert not np.any(changed & np.array(cert['certified']))
        report['metrics']['mean_perturbed_BA'] = float(np.mean(noise))
        equivalent(report, record['evaluation']); metrics[m] = report['metrics']; sizes[m] = len(fs)
        counts['outer_predictions'] += 1
    assert count == sizes['proposed'] == sizes['fgs_matched'] == sizes['sfrip_matched'] == sizes['fgs_uniform_matched'] == sizes['sfrip_uniform_matched']
    row = {'dataset': name, 'd': saved['d'], 'actual_counts': sizes, 'removed': 1-count/saved['d'], 'metrics': metrics}
    print(name, 'independent scalar5NN/query/selection audited', flush=True)
    return {'row': row, 'counts': dict(counts), 'artifact_sha256': digest(artifact)}


if __name__ == '__main__':
    summary = json.loads((FOLDER/'complete.json').read_text()); reports, counts = [], Counter()
    with ProcessPoolExecutor(max_workers=2) as executor:
        for f in as_completed([executor.submit(audit, r['dataset']) for r in summary['rows']]):
            r = f.result(); reports.append(r); counts.update(r['counts'])
    rows = sorted([r['row'] for r in reports], key=lambda r: r['dataset']); equivalent(rows, sorted(summary['rows'], key=lambda r: r['dataset']))
    means = {m: {k: float(np.mean([r['metrics'][m][k] for r in rows])) for k in rows[0]['metrics']['proposed']} for m in METHODS}
    equivalent(means, summary['means'])
    contrasts = {m: paired([r['metrics']['proposed']['clean_BA']-r['metrics'][m]['clean_BA'] for r in rows]) for m in METHODS if m != 'proposed'}
    equivalent(contrasts, summary['contrasts_clean_BA']); primary = ('fgs_matched', 'sfrip_matched')
    equivalent(holm({m: contrasts[m]['sign_flip_two_sided_p'] for m in primary}), summary['primary_descriptive_holm'])
    removed = float(np.mean([r['removed'] for r in rows])); equivalent(removed, summary['mean_removed'])
    gate = all(contrasts[m]['mean_pp'] >= .5 and contrasts[m]['wins'] >= 10 for m in primary) and removed >= .1 and contrasts['full']['mean_pp'] >= 0
    assert summary['pilot_gate_passed'] == gate
    output = {'passed': True, 'counts': dict(counts), 'artifact_hashes': {r['row']['dataset']: r['artifact_sha256'] for r in reports},
              'summary_sha256': digest(FOLDER/'complete.json'), 'auditor_sha256': digest(Path(__file__)), 'goal_achieved': False,
              'scope': 'independent_scalar5NN_votes_foldchoice_metrics_querybounds_and_shared_perturbations_not_confirmation'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(output, indent=2), encoding='utf-8'); print(output, flush=True)
