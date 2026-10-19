"""ALL15fold-local predictive metrics with same-grid recent weighted controls."""
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_multiscale_certified_metric_knn_pool import ROOT, FOLDER as PARENT, METHODS, COUNTS, policy_metrics, evaluate, digest
from soft_topk_knn_stage import GRID, PredictiveMetricStage, PredictionRegistry
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.knn_query import certify_knn_queries
from review_box_iterations import paired
from run_nested_joint_path import holm

FOLDER = ROOT/'results/soft-topk-knn-pilot'
WEIGHTED = {'proposed', 'unguarded_pool', 'raw_repair_pool', 'full', 'fgs_matched', 'sfrip_matched'}
INITIAL_METHODS = METHODS[:6]
MATCHED_METHODS = METHODS[6:]


def options(method):
    return [{'selector_index': si, 'target_index': ti, 'folds': []}
            for si in range(COUNTS[method]) for ti in range(len(GRID) if method in WEIGHTED else 1)]


def choose(rows):
    for row in rows:
        for k in ('clean_BA', 'balanced_logloss', 'actual_count'):
            row['mean_'+k] = float(np.mean([v[k] for v in row['folds']]))
    return min(range(len(rows)), key=lambda i: (-round(rows[i]['mean_clean_BA'], 12), round(rows[i]['mean_balanced_logloss'], 12),
                                               rows[i]['mean_actual_count'], rows[i]['selector_index'], rows[i]['target_index']))


def verified_parent(name):
    path = PARENT/(name+'.json'); audit = json.loads((PARENT/'complete-audit.json').read_text())
    assert audit['passed'] and audit['artifact_hashes'][name] == digest(path)
    parent = json.loads(path.read_text())
    for rel, sha in parent['fingerprints'].items(): assert digest(ROOT/rel) == sha, rel
    return parent


def fingerprints(name):
    files = [Path(__file__), ROOT/'scripts/soft_topk_knn_stage.py', ROOT/'scripts/capped_predictive_metric.py', ROOT/'scripts/capped_soft_topk_metric.py', ROOT/'scripts/soft_topk_metric_objective.py', ROOT/'scripts/predictive_target_knn_stage.py', ROOT/'scripts/soft_topk_predictive_metric.py',
             ROOT/'research/soft-topk-knn-pilot-protocol.md', PARENT/(name+'.json'), PARENT/'complete-audit.json',
             ROOT/'scripts/run_multiscale_certified_metric_knn_pool.py', ROOT/'scripts/run_certified_metric_knn_pool.py',
             ROOT/'results/predictive-target-metric-admission/complete-audit.json', ROOT/'results/majority-constrained-target-admission-v3/complete-audit.json', ROOT/'results/soft-topk-target-admission/complete-audit.json', ROOT/'data/processed'/(name+'.npz')]
    files += sorted((ROOT/'src/certified_frfs').glob('*.py'))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in files}


def score_options(method, bases, stage, registry):
    rows = options(method)
    for row in rows:
        metric = stage.metric(bases[row['selector_index']], row['target_index'], certified=method == 'proposed', weighted=method in WEIGHTED)
        prediction_id = registry.score(metric); report = registry.predictions[prediction_id]
        row['folds'] = [{**report['metrics'], 'actual_count': len(metric['features'])}]
        row['metric'] = metric; row['prediction_id'] = prediction_id
    return rows


def final_evaluation(A, labels, B, truth, metric):
    fs, w = metric['features'], np.array(metric['weights'])
    report = evaluate(A, labels, B, truth, fs, w); nominal = np.array(report['predictions'])
    if fs:
        embedded = np.zeros(A.shape[1]); embedded[fs] = w
        cert = certify_knn_queries(A, labels, B, nominal, fs, weights=embedded, k=5, allowed_enemies=2)
    else:
        cert = {'certified': [False]*len(B), 'scope': 'no_neighborhood_certificate_for_empty_prior'}
    correct = (nominal == truth) & np.array(cert['certified'])
    report['metrics']['query_certified_BA'] = float(np.mean([np.mean(correct[truth == c]) for c in np.unique(truth)]))
    noise = []
    for draw in range(16):
        rng = np.random.default_rng(20261060+draw)
        noisy_A = np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1)
        noisy_B = np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1)
        noise.append(evaluate(noisy_A, labels, noisy_B, truth, fs, w))
    report['metrics']['mean_perturbed_BA'] = float(np.mean([r['metrics']['clean_BA'] for r in noise]))
    return {'evaluation': report, 'query_certificate': cert, 'perturbation_evaluations': noise}


def run(name):
    threadpool_limits(1)
    parent = verified_parent(name); hashes = fingerprints(name); out = FOLDER/(name+'.json')
    if out.exists():
        previous = json.loads(out.read_text()); assert previous['fingerprints'] == hashes
        if previous['complete']: return
    X, y = load_dataset(name); ri, vi = np.array(parent['reference_indices']), np.array(parent['validation_indices'])
    raw, labels = X[ri], y[ri]; blocks = parent['candidate_reference_blocks']
    inners, stages, registries = [], [], []
    for fi, block in enumerate(blocks[:-1]):
        ca, cb = np.array(block['reference_indices']), np.array(block['validation_indices'])
        A, B = preprocess_fit(raw[ca], raw[cb]); stage = PredictiveMetricStage(A, labels[ca]); registry = PredictionRegistry(stage, B, labels[cb])
        bases = policy_metrics(A, block)
        rows = {m: score_options(m, bases[m], stage, registry) for m in INITIAL_METHODS}
        inners.append({'reference_indices': ca.tolist(), 'validation_indices': cb.tolist(), 'options': rows})
        stages.append(stage); registries.append(registry)
        print(name, 'inner', fi, 'target fits', len(stage.learners), 'guard checks', len(stage.certificates), 'predictions', len(registry.predictions), flush=True)
    combined, selected = {}, {}
    for m in INITIAL_METHODS:
        combined[m] = options(m)
        for i, row in enumerate(combined[m]): row['folds'] = [b['options'][m][i]['folds'][0] for b in inners]
        selected[m] = choose(combined[m])
    recipe_index = combined['proposed'][selected['proposed']]['selector_index']
    for block, record, stage, registry in zip(blocks[:-1], inners, stages, registries):
        count = len(policy_metrics(stage.A, block)['proposed'][recipe_index]['features'])
        record['matched_proposed_count'] = count; bases = policy_metrics(stage.A, block, count=count)
        for m in MATCHED_METHODS: record['options'][m] = score_options(m, bases[m], stage, registry)
        record.update(stage.export()); record['predictions'] = registry.predictions
    for m in MATCHED_METHODS:
        combined[m] = options(m)
        for i, row in enumerate(combined[m]): row['folds'] = [b['options'][m][i]['folds'][0] for b in inners]
        selected[m] = choose(combined[m])
    A, B = preprocess_fit(raw, X[vi]); stage = PredictiveMetricStage(A, labels)
    count = len(policy_metrics(A, blocks[-1])['proposed'][recipe_index]['features'])
    bases = policy_metrics(A, blocks[-1], count=count); results = {}
    for m in METHODS:
        row = combined[m][selected[m]]
        metric = stage.metric(bases[m][row['selector_index']], row['target_index'], certified=m == 'proposed', weighted=m in WEIGHTED)
        results[m] = {'option_index': selected[m], 'selector_index': row['selector_index'], 'target_index': row['target_index'],
                      'metric': metric, **final_evaluation(A, labels, B, y[vi], metric)}
    assert hashes == fingerprints(name)
    saved = {'dataset': name, 'd': X.shape[1], 'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(), 'fingerprints': hashes,
             'inner': inners, 'selection': {'options': combined, 'selected': selected},
             'outer': {'results': results, 'matched_proposed_count': count, **stage.export()},
             'complete': True, 'query_labels_used_for_selection': False, 'independent_audit_required': True,
             'scope': 'all15_development_dual_metric_families_equal_weighted_controls_not_confirmation'}
    out.write_text(json.dumps(saved, separators=(',', ':')), encoding='utf-8')
    print(name, 'COMPLETE', {m: round(results[m]['evaluation']['metrics']['clean_BA'], 6) for m in METHODS}, flush=True)


def summarize(names):
    rows = []; counts = Counter()
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); results = saved['outer']['results']
        assert saved['complete'] and not saved['query_labels_used_for_selection']
        sizes = {m: len(results[m]['metric']['features']) for m in METHODS}
        assert sizes['proposed'] == sizes['fgs_matched'] == sizes['sfrip_matched'] == sizes['fgs_uniform_matched'] == sizes['sfrip_uniform_matched']
        rows.append({'dataset': name, 'd': saved['d'], 'actual_counts': sizes, 'removed': 1-sizes['proposed']/saved['d'],
                     'metrics': {m: results[m]['evaluation']['metrics'] for m in METHODS}})
        for block in saved['inner']+[saved['outer']]:
            counts['target_fits'] += len(block['learners']); counts['guard_checks'] += len(block['certificate_checks'])
            counts['target_optimizer_failures'] += sum(not r['fit']['optimizer_success'] for r in block['learners'])
            counts['guard_rejections'] += sum(not r['accepted'] for r in block['certificate_checks'])
        counts['inner_folds'] += len(saved['inner']); counts['datasets'] += 1
    means = {m: {k: float(np.mean([r['metrics'][m][k] for r in rows])) for k in rows[0]['metrics']['proposed']} for m in METHODS}
    contrasts = {m: paired([r['metrics']['proposed']['clean_BA']-r['metrics'][m]['clean_BA'] for r in rows]) for m in METHODS if m != 'proposed'}
    primary = ('fgs_matched', 'sfrip_matched'); removed = float(np.mean([r['removed'] for r in rows]))
    gate = all(contrasts[m]['mean_pp'] >= .5 and contrasts[m]['wins'] >= 10 for m in primary) and removed >= .1 and contrasts['full']['mean_pp'] >= 0 and counts['target_optimizer_failures'] == 0
    summary = {'rows': rows, 'means': means, 'contrasts_clean_BA': contrasts, 'mean_removed': removed, 'counts': dict(counts),
               'pilot_gate_passed': bool(gate), 'primary_descriptive_holm': holm({m: contrasts[m]['sign_flip_two_sided_p'] for m in primary}),
               'goal_achieved': False, 'independent_audit_required': True, 'scope': 'all15_development_same_target_grid_controls_not_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(summary, indent=2), encoding='utf-8'); print({k: v for k, v in summary.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True); names = sorted(json.loads((PARENT/'complete-audit.json').read_text())['artifact_hashes'])
    with ProcessPoolExecutor(max_workers=2) as executor:
        for future in as_completed([executor.submit(run, name) for name in names]): future.result()
    summarize(names)
