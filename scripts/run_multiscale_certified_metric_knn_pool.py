"""The certified distance actually predicts: matched weighted5NN controls."""
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from run_certified_diverse_pool_pilot import ROOT, FOLDER as PARENT, BASE, digest
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.redistributed_loss import transfer_weights
from certified_frfs.capped_sparse import capped_sparse_projection
from certified_frfs.knn_query import certify_knn_queries
from review_box_iterations import paired
from run_nested_joint_path import holm

FOLDER = ROOT/'results/multiscale-certified-metric-knn-pool'
METHODS = ('proposed', 'unguarded_pool', 'full', 'fgs_native', 'sfrip_native', 'raw_repair_pool',
           'fgs_matched', 'sfrip_matched', 'fgs_uniform_matched', 'sfrip_uniform_matched')
COUNTS = {'proposed': 47, 'unguarded_pool': 47, 'full': 1, 'fgs_native': 9, 'sfrip_native': 6, 'raw_repair_pool': 47,
          'fgs_matched': 9, 'sfrip_matched': 6, 'fgs_uniform_matched': 9, 'sfrip_uniform_matched': 6}


from run_recent_multiscale_pool_admission import recipes as scale_recipes, admission as scale_admission
SCALE_POOL = ROOT/'results/recent-multiscale-pool-admission'


def multiscale_parent(name, old, raw, labels):
    artifact = SCALE_POOL/(name+'.json')
    audit = json.loads((SCALE_POOL/'complete-audit.json').read_text())
    assert audit['passed'] and audit['artifact_hashes'][name] == digest(artifact)
    outer = json.loads(artifact.read_text())
    assert outer['reference_indices'] == old['reference_indices']
    for rel, sha in outer['fingerprints'].items(): assert digest(ROOT/rel) == sha, rel
    own = json.loads((ROOT/'results/crossfit-label-loss-recent-pilot'/(name+'.json')).read_text())
    result = {**old, 'inner': []}
    for fi, block in enumerate(old['inner']):
        ca = np.array(block['reference_indices'])
        candidates = scale_recipes(block['paths'], own['inner'][fi]['stage']['candidate']['features'], raw.shape[1])
        pool = scale_admission(raw[ca], labels[ca], candidates, block['pool'])
        result['inner'].append({**block, 'recipes': candidates, 'pool': pool})
        print(name, 'inner', fi, 'multiscale pool', sum(r['accepted'] for r in pool), '/', len(pool), flush=True)
    result['outer'] = {**old['outer'], 'recipes': outer['recipes'], 'pool': outer['pool']}
    return result


def evaluate(A, y, B, truth, fs, weights):
    classes = np.unique(y)
    if fs:
        distance = cdist(B[:, fs]*weights, A[:, fs]*weights, 'cityblock')
        nn = np.argsort(distance, axis=1, kind='stable')[:, :5]
        votes = np.stack([(y[nn] == c).sum(axis=1) for c in classes], axis=1)
        probabilities = (votes+1)/(5+len(classes))
        pred = classes[votes.argmax(axis=1)]
    else:
        probabilities = np.tile([(np.sum(y == c)+1)/(len(y)+len(classes)) for c in classes], (len(B), 1))
        pred = classes[probabilities.argmax(axis=1)]
        nn = np.zeros((len(B), 0), int)
    def balanced(x): return float(np.mean([np.mean(x[truth == c]) for c in np.unique(truth)]))
    own = np.searchsorted(classes, truth)
    metrics = {'clean_BA': balanced(pred == truth), 'balanced_logloss': balanced(-np.log(probabilities[np.arange(len(B)), own])),
               'macro_F1': float(f1_score(truth, pred, average='macro', zero_division=0))}
    return {'predictions': pred.tolist(), 'probabilities': probabilities.tolist(), 'neighbor_indices': nn.tolist(), 'metrics': metrics}


def choose(rows):
    for row in rows:
        for k in ('clean_BA', 'balanced_logloss', 'actual_count'):
            row['mean_'+k] = float(np.mean([v[k] for v in row['folds']]))
    return min(range(len(rows)), key=lambda i: (-round(rows[i]['mean_clean_BA'], 12), round(rows[i]['mean_balanced_logloss'], 12), rows[i]['mean_actual_count'], i))


def policy_metrics(A, block, count=None):
    d = A.shape[1]; full = list(range(d)); uniform = np.ones(d)/np.sqrt(d)
    by_support = {tuple(r['features']): r for r in block['pool']}
    def transferred(fs):
        fs = sorted(fs)
        weights = uniform if len(fs) == d else capped_sparse_projection(transfer_weights(A, fs, uniform)/np.sqrt(d), len(fs), 2/d)*np.sqrt(d)
        return {'features': fs, 'weights': weights.tolist(), 'origin': 'ordinary_transferred_capped_metric'}
    output = {m: [] for m in METHODS}
    for recipe in block['recipes']:
        r = by_support[tuple(recipe['features'])]
        metric = r['selected_metric'] if r['accepted'] else {'features': full, 'weights': uniform.tolist()}
        fs = metric['features']
        # Admission stores full-dimensional embedded weights.
        wide = np.array(metric['weights'])
        output['proposed'].append({'features': fs, 'weights': wide[fs].tolist(), 'origin': 'admitted_metric' if r['accepted'] else 'explicit_full_fallback'})
        ordinary = r['ordinary']; fs = ordinary['features']
        info = r['repair']
        rw = info['weights'] if info is not None and 'weights' in info else np.array(ordinary['weights'])[fs].tolist()
        output['raw_repair_pool'].append({'features': fs, 'weights': rw, 'origin': 'raw_normalized_repair_metric' if info is not None and 'weights' in info else 'ordinary_uncensored_metric'})
        output['unguarded_pool'].append({'features': fs, 'weights': np.array(ordinary['weights'])[fs].tolist(), 'origin': 'ordinary_uncensored_metric'})
    output['full'] = [{'features': full, 'weights': uniform.tolist(), 'origin': 'uniform_full'}]
    for method, key in (('fgs', 'fgs2023'), ('sfrip', 'sfrip2025')):
        for p in block['paths'][key]:
            native = sorted(p['native']['features'])
            output[method+'_native'].append({'features': native, 'weights': np.ones(len(native)).tolist(), 'origin': 'uniform_native'})
            if count is not None:
                fs = sorted(p['ranking'][:count])
                output[method+'_matched'].append(transferred(fs))
                output[method+'_uniform_matched'].append({'features': fs, 'weights': np.ones(len(fs)).tolist(), 'origin': 'uniform_count_matched'})
    return {m: v for m, v in output.items() if v}


def fingerprints(name):
    files = [Path(__file__), ROOT/'research/multiscale-certified-metric-knn-pool-protocol.md', PARENT/(name+'.json'), PARENT/'complete-audit.json', ROOT/'data/processed'/(name+'.npz')]
    files += [SCALE_POOL/(name+'.json'), SCALE_POOL/'complete-audit.json', ROOT/'scripts/run_recent_multiscale_pool_admission.py', ROOT/'scripts/prepare_multiscale_metric_knn_pilot.py']
    files += sorted((ROOT/'src/certified_frfs').glob('*.py'))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in files}


def run(name):
    threadpool_limits(1)
    hashes = fingerprints(name); out = FOLDER/(name+'.json')
    if out.exists():
        old = json.loads(out.read_text()); assert old['fingerprints'] == hashes
        if old['complete']: return
    audit = json.loads((PARENT/'complete-audit.json').read_text()); source = PARENT/(name+'.json')
    assert audit['passed'] and audit['artifact_hashes'][name] == digest(source)
    parent = json.loads(source.read_text())
    for rel, sha in parent['fingerprints'].items(): assert digest(ROOT/rel) == sha, rel
    X, y = load_dataset(name); ri, vi = np.array(parent['reference_indices']), np.array(parent['validation_indices'])
    raw, labels = X[ri], y[ri]
    parent = multiscale_parent(name, parent, raw, labels)
    inners = []; combined = {m: [{'selector_index': i, 'folds': []} for i in range(COUNTS[m])] for m in METHODS}
    for block in parent['inner']:
        ca, cb = np.array(block['reference_indices']), np.array(block['validation_indices'])
        A, B = preprocess_fit(raw[ca], raw[cb]); policies = policy_metrics(A, block)
        rows = {}
        for m in METHODS[:6]:
            rows[m] = []
            for i, metric in enumerate(policies[m]):
                evaluation = evaluate(A, labels[ca], B, labels[cb], metric['features'], np.array(metric['weights']))
                fold = {**evaluation['metrics'], 'actual_count': len(metric['features'])}
                rows[m].append({'metric': metric, 'evaluation': evaluation}); combined[m][i]['folds'].append(fold)
        inners.append({'reference_indices': ca.tolist(), 'validation_indices': cb.tolist(), 'options': rows})
    selected = {m: choose(combined[m]) for m in METHODS[:6]}
    for block, record in zip(parent['inner'], inners):
        ca, cb = np.array(block['reference_indices']), np.array(block['validation_indices'])
        A, B = preprocess_fit(raw[ca], raw[cb])
        count = len(record['options']['proposed'][selected['proposed']]['metric']['features'])
        record['matched_proposed_count'] = count
        policies = policy_metrics(A, block, count=count)
        for m in METHODS[6:]:
            record['options'][m] = []
            for i, metric in enumerate(policies[m]):
                evaluation = evaluate(A, labels[ca], B, labels[cb], metric['features'], np.array(metric['weights']))
                combined[m][i]['folds'].append({**evaluation['metrics'], 'actual_count': len(metric['features'])})
                record['options'][m].append({'metric': metric, 'evaluation': evaluation})
    selected.update({m: choose(combined[m]) for m in METHODS[6:]})
    A, B = preprocess_fit(raw, X[vi]); initial = policy_metrics(A, parent['outer'])
    count = len(initial['proposed'][selected['proposed']]['features'])
    policies = policy_metrics(A, parent['outer'], count=count); results = {}
    for m in METHODS:
        metric = policies[m][selected[m]]; fs, w = metric['features'], np.array(metric['weights'])
        evaluation = evaluate(A, labels, B, y[vi], fs, w)
        nominal = np.array(evaluation['predictions'])
        if fs:
            embedded = np.zeros(X.shape[1]); embedded[fs] = w
            cert = certify_knn_queries(A, labels, B, nominal, fs, weights=embedded, k=5, allowed_enemies=2)
        else:
            cert = {'certified': [False]*len(B), 'scope': 'no_neighborhood_certificate_for_empty_prior'}
        correct = (nominal == y[vi]) & np.array(cert['certified'])
        evaluation['metrics']['query_certified_BA'] = float(np.mean([np.mean(correct[y[vi] == c]) for c in np.unique(y[vi])]))
        noises = []
        for draw in range(16):
            rng = np.random.default_rng(20261060+draw)
            noisy_A = np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1)
            noisy_B = np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1)
            noises.append(evaluate(noisy_A, labels, noisy_B, y[vi], fs, w))
        evaluation['metrics']['mean_perturbed_BA'] = float(np.mean([v['metrics']['clean_BA'] for v in noises]))
        results[m] = {'selector_index': selected[m], 'metric': metric, 'evaluation': evaluation,
                      'query_certificate': cert, 'perturbation_evaluations': noises}
    assert hashes == fingerprints(name)
    saved = {'dataset': name, 'd': X.shape[1], 'fingerprints': hashes, 'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(),
             'candidate_reference_blocks': [{k: block[k] for k in ('reference_indices', 'validation_indices', 'recipes', 'pool', 'paths')} for block in parent['inner']]+[{k: parent['outer'][k] for k in ('recipes', 'pool', 'paths')}],
             'inner': inners, 'selection': {'options': combined, 'selected': selected}, 'outer': {'results': results, 'matched_proposed_count': count},
             'complete': True, 'query_labels_used_for_selection': False, 'independent_audit_required': True,
             'scope': 'all15_exposed_development_actual_certified_metric_5NN_not_confirmation'}
    out.write_text(json.dumps(saved, separators=(',', ':')), encoding='utf-8')
    print(name, 'COMPLETE', {m: round(results[m]['evaluation']['metrics']['clean_BA'], 6) for m in METHODS}, flush=True)


def summarize(names):
    rows = []
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); r = saved['outer']['results']
        sizes = {m: len(r[m]['metric']['features']) for m in METHODS}
        assert sizes['proposed'] == sizes['fgs_matched'] == sizes['sfrip_matched'] == sizes['fgs_uniform_matched'] == sizes['sfrip_uniform_matched']
        rows.append({'dataset': name, 'd': saved['d'], 'actual_counts': sizes, 'removed': 1-sizes['proposed']/saved['d'],
                     'metrics': {m: r[m]['evaluation']['metrics'] for m in METHODS}})
    metrics = rows[0]['metrics']['proposed'].keys()
    means = {m: {k: float(np.mean([r['metrics'][m][k] for r in rows])) for k in metrics} for m in METHODS}
    contrasts = {m: paired([r['metrics']['proposed']['clean_BA']-r['metrics'][m]['clean_BA'] for r in rows]) for m in METHODS if m != 'proposed'}
    primary = ('fgs_matched', 'sfrip_matched'); removed = float(np.mean([r['removed'] for r in rows]))
    gate = all(contrasts[m]['mean_pp'] >= .5 and contrasts[m]['wins'] >= 10 for m in primary) and removed >= .1 and contrasts['full']['mean_pp'] >= 0
    report = {'rows': rows, 'means': means, 'contrasts_clean_BA': contrasts, 'mean_removed': removed, 'pilot_gate_passed': bool(gate),
              'primary_descriptive_holm': holm({m: contrasts[m]['sign_flip_two_sided_p'] for m in primary}),
              'goal_achieved': False, 'independent_audit_required': True, 'scope': 'all15_development_certified_metric_classifier_not_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(report, indent=2), encoding='utf-8'); print({k: v for k, v in report.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True)
    names = sorted(json.loads((PARENT/'complete-audit.json').read_text())['artifact_hashes'])
    with ProcessPoolExecutor(max_workers=2) as executor:
        for f in as_completed([executor.submit(run, name) for name in names]): f.result()
    summarize(names)
