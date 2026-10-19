"""Nested original-FRS supports and tuned native/matched recent baselines."""
import json
import time
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT, digest, holm
from compare_trajectory_recent import GRIDS
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.nested_teacher_regularized import inner_splits
from certified_frfs.fuzzy_interaction_tree import interaction_tree
from certified_frfs.multiknot_forest import fit_multiknot_forest, scores, box_margins
from certified_frfs.adaptive_multiknot_stage import MODEL_GRID
from certified_frfs.safeguarded_crossfit_stage import fit_safeguarded_crossfit_stage
from certified_frfs.recent_grid_fast import fgs_paths_grid, sfrip_paths_grid
from review_box_iterations import paired

FOLDER = ROOT/'results/safeguarded-crossfit-recent-pilot'
METHODS = ('proposed', 'full', 'fgs_native', 'fgs_matched', 'sfrip_native', 'sfrip_matched')
METRICS = ('clean_BA', 'balanced_logloss', 'macro_F1', 'certified_BA')


def metric_scores(X, truth, fit, *, certify=False):
    value = scores(X, fit)
    classes = np.array(fit['classes'])
    own = np.searchsorted(classes, truth)
    pred = classes[value.argmax(axis=1)]
    loss = logsumexp(value, axis=1)-value[np.arange(len(X)), own]
    def balanced(array):
        return float(np.mean([np.mean(array[truth == c]) for c in np.unique(truth)]))
    metrics = {'clean_BA': balanced(pred == truth), 'balanced_logloss': balanced(loss),
               'macro_F1': float(f1_score(truth, pred, average='macro', zero_division=0))}
    result = {'scores': value.tolist(), 'predictions': pred.tolist(), 'metrics': metrics}
    if certify:
        oracle = box_margins(X, truth, fit, eps=.01, witnesses=True)
        pair = oracle['pair_margins'].copy()
        pair[np.arange(len(X)), own] = 0.
        result['pair_margins'] = pair.tolist()
        metrics['certified_BA'] = balanced(oracle['margins'] > 1e-10)
    return result


class Registry:
    def __init__(self, A, train, B=None, truth=None):
        self.A, self.train, self.B, self.truth = A, train, B, truth
        self.trees, self.models, self.evaluations, self.lookup = {}, [], {}, {}

    def fit(self, features, model_index):
        fs = tuple(sorted(features))
        key = (fs, model_index)
        if key not in self.lookup:
            config = MODEL_GRID[model_index]
            if fs:
                if fs not in self.trees:
                    local = interaction_tree(self.A[:, fs], self.train)
                    self.trees[fs] = {'features': list(fs), 'local_tree': local,
                                     'edges': [sorted([fs[u], fs[v]]) for u, v in local['edges']]}
                model = fit_multiknot_forest(self.A, self.train, list(fs), self.trees[fs]['edges'], **config)
            else:
                classes, counts = np.unique(self.train, return_counts=True)
                q, d, m = len(classes), self.A.shape[1], config['knot_count']
                model = {'knots': np.linspace(0, 1, m).tolist(), 'classes': classes.tolist(), 'features': [], 'edges': [],
                    'coef': np.zeros((q, d, m)).tolist(), 'pair_coef': np.zeros((q, 0, m, m)).tolist(),
                    'intercept': np.log(counts/counts.sum()).tolist(), **config,
                    'optimizer_iterations': 0, 'optimizer_converged': True, 'class_balanced': False,
                    'teacher_guards_enforced': False, 'design_columns': 0,
                    'scope': 'explicit_empty_native_support_unweighted_training_prior_no_restoration'}
            mid = len(self.models)
            self.lookup[key] = mid
            self.models.append(model)
            if self.B is not None:
                self.evaluations[str(mid)] = metric_scores(self.B, self.truth, model)
        return self.lookup[key]

    def export(self):
        return {'trees': list(self.trees.values()), 'models': self.models, 'evaluations': self.evaluations}


def method_supports(stage, paths, d):
    count = stage['actual_count']
    return {'proposed': [stage['candidate']['features']], 'full': [list(range(d))],
            'fgs_native': [p['native']['features'] for p in paths['fgs2023']],
            'fgs_matched': [p['ranking'][:count] for p in paths['fgs2023']],
            'sfrip_native': [p['native']['features'] for p in paths['sfrip2025']],
            'sfrip_matched': [p['ranking'][:count] for p in paths['sfrip2025']]}


def path_grid(A, train):
    start = time.perf_counter()
    fgs = fgs_paths_grid(A, train, GRIDS['fgs2023'])
    fgs_seconds = time.perf_counter()-start
    start = time.perf_counter()
    sfrip = sfrip_paths_grid(A, train, GRIDS['sfrip2025'])
    return {'fgs2023': fgs, 'sfrip2025': sfrip}, {'fgs_grid_seconds': fgs_seconds, 'sfrip_grid_seconds': time.perf_counter()-start}


def options():
    counts = {'proposed': 1, 'full': 1, 'fgs_native': 9, 'fgs_matched': 9, 'sfrip_native': 6, 'sfrip_matched': 6}
    return {m: [{'selector_index': si, 'model_index': mi, 'folds': []}
                for si in range(counts[m]) for mi in range(len(MODEL_GRID))] for m in METHODS}


def choose(rows):
    for row in rows:
        row['mean_clean_BA'] = float(np.mean([v['clean_BA'] for v in row['folds']]))
        row['mean_actual_count'] = float(np.mean([v['actual_count'] for v in row['folds']]))
    return min(range(len(rows)), key=lambda i: (-round(rows[i]['mean_clean_BA'], 12), rows[i]['mean_actual_count'],
                                               rows[i]['selector_index'], rows[i]['model_index']))


def fingerprints(name):
    paths = [Path(__file__), ROOT/'research/adaptive-multiknot-recent-pilot-protocol.md', ROOT/'research/projected-label-loss-next-protocol.md', ROOT/'research/crossfit-label-loss-pilot-protocol.md', ROOT/'research/safeguarded-crossfit-recent-pilot-protocol.md', ROOT/'scripts/prepare_safeguarded_crossfit_recent_pilot.py', ROOT/'scripts/prepare_repaired_crossfit_recent_pilot.py', ROOT/'scripts/prepare_crossfit_label_loss_pilot.py',
             ROOT/'results/adaptive-multiknot-recent-pilot'/(name+'.json'),
             ROOT/'results/adaptive-multiknot-recent-pilot'/(name+'-audit.json'),
             ROOT/'results/adaptive-multiknot-recent-pilot/complete-audit.json',
             ROOT/'scripts/prepare_projected_label_loss_recent_pilot.py',
             ROOT/'scripts/compare_trajectory_recent.py', ROOT/'scripts/run_nested_joint_path.py',
             ROOT/'results/predictive-iteration41'/(name+'-s101-f0.json'), ROOT/'data/processed'/(name+'.npz')]
    paths += sorted((ROOT/'src/certified_frfs').glob('*.py'))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}



def verified_cached_paths(name, inner_index, A, train):
    base = ROOT/'results/adaptive-multiknot-recent-pilot'
    artifact = base/(name+'.json')
    saved = json.loads(artifact.read_text())
    audit = json.loads((base/(name+'-audit.json')).read_text())
    complete = json.loads((base/'complete-audit.json').read_text())
    assert complete['passed'] and complete['artifact_hashes'][name] == digest(artifact)
    assert audit['fingerprints']['artifact_sha256'] == digest(artifact)
    assert audit['fingerprints']['auditor_sha256'] == digest(ROOT/'scripts/audit_adaptive_multiknot_recent_pilot.py')
    for rel, sha in saved['fingerprints'].items():
        assert digest(ROOT/rel) == sha, rel
    X, y = load_dataset(name)
    ri = np.array(saved['reference_indices'])
    if inner_index is None:
        block, ref, labels = saved['outer'], X[ri], y[ri]
    else:
        block = saved['inner'][inner_index]
        a = np.array(block['reference_indices'])
        ref, labels = X[ri[a]], y[ri[a]]
    expected, _ = preprocess_fit(ref, ref)
    np.testing.assert_array_equal(A, expected)
    np.testing.assert_array_equal(train, labels)
    costs = {key: block['costs'][key] for key in ('fgs_grid_seconds', 'sfrip_grid_seconds')}
    costs.update({'baseline_paths_reused': True, 'standalone_selector_time_measured_here': False})
    return block['paths'], costs

def run(name):
    threadpool_limits(1)
    hashes = fingerprints(name)
    out, inners = FOLDER/(name+'.json'), []
    if out.exists():
        old = json.loads(out.read_text())
        assert old['fingerprints'] == hashes
        if old['complete']:
            return
        inners = old['inner']
    X, y = load_dataset(name)
    original = json.loads((ROOT/'results/predictive-iteration41'/(name+'-s101-f0.json')).read_text())
    ri, vi = np.array(original['train_indices']), np.array(original['test_indices'])
    raw, labels = X[ri], y[ri]
    splits = inner_splits(labels, seed=20261033)
    for fi, (ca, cb) in enumerate(splits):
        if fi < len(inners):
            continue
        start = time.perf_counter()
        stage = fit_safeguarded_crossfit_stage(raw[ca], labels[ca])
        A, B = preprocess_fit(raw[ca], raw[cb])
        stage_seconds = time.perf_counter()-start
        print(name, 'inner', fi, 'stage', stage['actual_count'], '/', X.shape[1], 'seconds', round(stage_seconds, 2), flush=True)
        paths, costs = verified_cached_paths(name, fi, A, labels[ca])
        registry = Registry(A, labels[ca], B, labels[cb])
        supports = method_supports(stage, paths, X.shape[1])
        rows = options()
        fit_start = time.perf_counter()
        for m in METHODS:
            for row in rows[m]:
                fs = supports[m][row['selector_index']]
                mid = registry.fit(fs, row['model_index'])
                row['folds'].append({'model_id': mid, 'features': sorted(fs), 'actual_count': len(fs),
                                     'clean_BA': registry.evaluations[str(mid)]['metrics']['clean_BA']})
        costs.update({'stage_seconds': stage_seconds, 'classifier_grid_seconds': time.perf_counter()-fit_start})
        inners.append({'reference_indices': ca.tolist(), 'validation_indices': cb.tolist(), 'stage': stage,
                       'paths': paths, 'options': rows, 'costs': costs, **registry.export()})
        out.write_text(json.dumps({'dataset': name, 'raw_n': len(X), 'd': X.shape[1], 'fingerprints': hashes,
            'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(), 'inner': inners, 'complete': False,
            'scope': 'exposed_development_first_raw_outer_fold_original_FRS_safeguarded_repairs_and_tuned_recent'}, separators=(',', ':')), encoding='utf-8')
        print(name, 'inner', fi, 'all recent/model options', len(registry.models), 'completed', flush=True)
    combined = options()
    for m in METHODS:
        for i, row in enumerate(combined[m]):
            row['folds'] = [block['options'][m][i]['folds'][0] for block in inners]
    selected = {m: choose(combined[m]) for m in METHODS}
    start = time.perf_counter()
    stage = fit_safeguarded_crossfit_stage(raw, labels)
    A, B = preprocess_fit(raw, X[vi])
    stage_seconds = time.perf_counter()-start
    print(name, 'outer stage', stage['actual_count'], '/', X.shape[1], 'seconds', round(stage_seconds, 2), flush=True)
    paths, costs = verified_cached_paths(name, None, A, labels)
    supports = method_supports(stage, paths, X.shape[1])
    registry = Registry(A, labels)
    results = {}
    fit_start = time.perf_counter()
    for m in METHODS:
        option = combined[m][selected[m]]
        fs = supports[m][option['selector_index']]
        mid = registry.fit(fs, option['model_index'])
        results[m] = {'option_index': selected[m], 'selector_index': option['selector_index'],
                      'model_index': option['model_index'], 'features': sorted(fs), 'model_id': mid,
                      'evaluation': metric_scores(B, y[vi], registry.models[mid], certify=True)}
    costs.update({'stage_seconds': stage_seconds, 'selected_classifier_and_evaluation_seconds': time.perf_counter()-fit_start})
    assert fingerprints(name) == hashes
    out.write_text(json.dumps({'dataset': name, 'raw_n': len(X), 'd': X.shape[1], 'fingerprints': hashes,
        'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(), 'inner': inners,
        'selection': {'options': combined, 'selected': selected},
        'outer': {'stage': stage, 'paths': paths, 'costs': costs, 'results': results, **registry.export()},
        'complete': True, 'query_labels_used_for_selection': False,
        'scope': 'exposed_development_first_raw_outer_fold_original_FRS_safeguarded_repairs_and_tuned_recent'}, separators=(',', ':')), encoding='utf-8')
    print(name, 'COMPLETE', {m: round(results[m]['evaluation']['metrics']['clean_BA'], 6) for m in METHODS}, flush=True)


def summarize(names):
    rows, counts = [], Counter()
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text())
        assert saved['complete'] and not saved['query_labels_used_for_selection']
        results = saved['outer']['results']
        metrics = {m: results[m]['evaluation']['metrics'] for m in METHODS}
        sizes = {m: len(results[m]['features']) for m in METHODS}
        assert sizes['proposed'] == sizes['fgs_matched'] == sizes['sfrip_matched']
        counts['datasets'] += 1
        counts['outer_folds'] += 1
        counts['inner_folds'] += len(saved['inner'])
        counts['outer_count_matches'] += 2
        counts['selected_full_proposed'] += sizes['proposed'] == saved['d']
        counts['empty_native_outputs'] += sizes['fgs_native'] == 0
        counts['empty_native_outputs'] += sizes['sfrip_native'] == 0
        for block in saved['inner']+[saved['outer']]:
            for repair in block['stage']['path']['repair_blocks']:
                counts['metric_repair_attempts'] += len(repair['attempts'])
                counts['metric_solver_limits'] += sum(r['fit']['solver_status'] == 1 for r in repair['attempts'])
                counts['accepted_metric_repairs'] += sum(r['fit']['status'] == 'original_certificate_passed' for r in repair['attempts'])
                counts['rejected_metric_repairs'] += sum(r['fit']['status'] != 'original_certificate_passed' for r in repair['attempts'])
                counts['failed_metric_repair_steps'] += repair['attempts'][-1]['fit']['status'] != 'original_certificate_passed'
            counts['teacher_optimizer_caps'] += block['stage']['teacher_selection']['optimizer_caps']
            counts['downstream_optimizer_caps'] += sum(not fit['optimizer_converged'] for fit in block['models'])
            counts['downstream_unique_fits'] += len(block['models'])
            counts['teacher_fits'] += 1+sum(len(f['models']) for f in block['stage']['teacher_selection']['folds'])
        rows.append({'dataset': name, 'metrics': metrics, 'actual_counts': sizes, 'd': saved['d'],
                     'removed': 1-sizes['proposed']/saved['d']})
    assert counts['datasets'] == 15 and counts['inner_folds'] == 44
    means = {m: {k: float(np.mean([r['metrics'][m][k] for r in rows])) for k in METRICS} for m in METHODS}
    contrasts = {m: paired([r['metrics']['proposed']['clean_BA']-r['metrics'][m]['clean_BA'] for r in rows])
                 for m in METHODS if m != 'proposed'}
    primary = ('fgs_matched', 'sfrip_matched')
    corrected = holm({m: contrasts[m]['sign_flip_two_sided_p'] for m in primary})
    removed = float(np.mean([r['removed'] for r in rows]))
    gate = all(contrasts[m]['mean_pp'] >= .5 and contrasts[m]['wins'] >= 10 for m in primary)
    gate = gate and contrasts['full']['mean_pp'] >= 0 and removed >= .1
    gate = gate and counts['teacher_optimizer_caps'] == counts['downstream_optimizer_caps'] == 0
    result = {'rows': rows, 'means': means, 'counts': dict(counts), 'contrasts_clean_BA': contrasts,
              'primary_descriptive_holm': corrected, 'mean_removed': removed, 'pilot_gate_passed': bool(gate),
              'goal_achieved': False, 'independent_audit_required': True,
              'scope': 'all15_first_raw_outer_folds_inner_selected_adaptive_reducts_not_confirmation_or_FD_FCE'}
    (FOLDER/'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({k: v for k, v in result.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((ROOT/'results/development-sample-depth-audit.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            future.result()
    summarize(names)
