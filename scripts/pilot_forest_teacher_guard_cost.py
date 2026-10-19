"""Explicit counterfactual on identical supports; never certified-task deployment."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import numpy as np
from threadpoolctl import threadpool_limits
from review_box_iterations import ROOT, paired
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.forest_teacher_guard_ablation import fit_forest_logistic
from certified_frfs.forest_fuzzy_classifier import forest_scores, forest_box_margins
from certified_frfs.triangular_classifier import triangular_box_margins
from certified_frfs.forest_penalty_selection import balanced_accuracy

SOURCE = ROOT/'results/nested-crossfit-classification-rounded'
SELECTION = ROOT/'results/forest-penalty-selection'
COMPACT = ROOT/'results/forest-compact-transfer'
AWARE = ROOT/'results/forest-aware-support'
FOLDER = ROOT/'results/forest-teacher-guard-cost'


def raw_model(fit, classes):
    edges = fit.get('edges', [])
    pair = np.asarray(fit.get('pair_coef', np.zeros((len(classes), 0, 3, 3))))
    return fit['coef'], fit['intercept'], edges, pair.reshape(len(classes), len(edges), 3, 3)


def run(name):
    threadpool_limits(1)
    paths = [f/(name+'.json') for f in (SOURCE, SELECTION, COMPACT, AWARE)]
    source, selected, compact, aware = [json.loads(p.read_text()) for p in paths]
    X, y = load_dataset(name); records = []; counter = 0
    for i, split in enumerate(source['splits']):
        ri = np.array(split['reference_indices']); raw, labels = X[ri], y[ri]
        for j, inner in enumerate(split['inner']):
            r, full, old = aware['records'][counter], selected['records'][counter], compact['records'][counter]; counter += 1
            for item in (r, full, old):
                assert item['outer_reference_indices'] == ri.tolist() and item['reference_indices'] == inner['reference_indices'] and item['validation_indices'] == inner['validation_indices']
            a, b = np.array(inner['reference_indices']), np.array(inner['validation_indices'])
            A, B = preprocess_fit(raw[a], raw[b]); yt, truth = labels[a], labels[b]
            stage = inner['by_C']['1.0']['stage']; classes = np.array(stage['teacher_classes'])
            key = r['selected_option']; edges = full['tree']['edges'] if key != 'additive' else []
            guarded = r['fit'] if key != 'additive' else old['policy']['fit']
            fit = fit_forest_logistic(A, yt, stage['teacher_coef'], stage['teacher_intercept'], classes,
                r['features'], edges, C=1., pair_penalty=float(key) if key != 'additive' else 1., enforce_teacher_guards=False)
            deployed = fit if fit['feasible'] else guarded
            W, bias, retained_edges, V = raw_model(deployed, classes)
            original = triangular_box_margins(A, yt, stage['teacher_coef'], stage['teacher_intercept'], classes)
            margins = forest_box_margins(A, yt, W, bias, classes, retained_edges, V)
            protected = original > 1e-10
            retained = protected & (margins >= .5*original-1e-8) & (margins > 1e-10)
            per_class = [{'class': int(c), 'protected': int(np.sum(protected & (yt == c))),
                'retained': int(np.sum(retained & (yt == c)))} for c in classes]
            rng = np.random.default_rng(r['query_seed'])
            queries = [B]+[np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1) for _ in range(3)]
            values = [balanced_accuracy(forest_scores(Q, W, bias, retained_edges, V), classes, truth) for Q in queries]
            records.append({'outer_reference_indices': ri.tolist(), 'reference_indices': a.tolist(), 'validation_indices': b.tolist(),
                'features': r['features'], 'selected_option': key, 'fit': fit, 'fallback': not fit['feasible'],
                'teacher_guards_enforced': False, 'actual_full_teacher_guards_verified': bool(np.all(retained[protected])),
                'original_margins': original.tolist(), 'deployed_margins': margins.tolist(), 'per_class_retention': per_class,
                'query_seed': r['query_seed'], 'metrics': {'clean': values[0], 'noise': float(np.mean(values[1:]))},
                'guarded_metrics': r['metrics'], 'removed': r['removed'],
                'scope': 'explicit_task_guard_counterfactual_on_fixed_FRS_certified_support_not_new_certified_classifier'})
            print(name, i, j, key, 'gain_pp', 100*(values[0]-r['metrics']['clean']), 'full_guard', records[-1]['actual_full_teacher_guards_verified'], flush=True)
    assert counter == len(aware['records'])
    files = paths+[Path(__file__), ROOT/'research/forest-teacher-guard-cost-protocol.md', ROOT/'data/processed'/(name+'.npz')]
    files += [ROOT/'src/certified_frfs'/f for f in ('forest_teacher_guard_ablation.py', 'forest_fuzzy_classifier.py',
        'triangular_classifier.py', 'validated_pool.py')]
    (FOLDER/(name+'.json')).write_text(json.dumps({'dataset': name, 'records': records,
        'fingerprints': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}, indent=2), encoding='utf-8')


def summarize(names):
    rows = []; counts = Counter()
    for name in names:
        records = json.loads((FOLDER/(name+'.json')).read_text())['records']
        counts['references'] += len(records); counts['numerical_fit_failures'] += sum(r['fallback'] for r in records)
        counts['full_teacher_guard_violations'] += sum(not r['actual_full_teacher_guards_verified'] for r in records)
        counts['references_below_90percent_any_protected_class'] += sum(any(c['protected'] and c['retained'] < int(np.ceil(.9*c['protected']-1e-12)) for c in r['per_class_retention']) for r in records)
        rows.append({'dataset': name, 'removed': float(np.mean([r['removed'] for r in records])),
            **{m: {metric: float(np.mean([r[field][metric] for r in records])) for metric in ('clean', 'noise')}
                for m, field in (('unguarded', 'metrics'), ('guarded', 'guarded_metrics'))}})
    assert len(rows) == 15 and counts['references'] == 128
    gain = paired([r['unguarded']['clean']-r['guarded']['clean'] for r in rows])
    result = {'rows': rows, 'counts': dict(counts), 'clean_gain': gain,
        'noise_gain': paired([r['unguarded']['noise']-r['guarded']['noise'] for r in rows]),
        'gate_passed': gain['mean_pp'] >= .5 and counts['numerical_fit_failures']/128 <= .1,
        'scope': 'explicit_classifier_guard_cost_diagnostic_not_certified_method_or_SOTA_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({k: v for k, v in result.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    assert (AWARE/'complete-audit.json').exists()
    FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]): future.result()
    summarize(names)
