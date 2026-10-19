"""Fixed certified .8-policy supports, matched additive and forest classifiers."""
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
from certified_frfs.centered_forest_fuzzy_logistic_2d import fit_forest_logistic
from certified_frfs.forest_fuzzy_classifier import forest_scores
from certified_frfs.forest_penalty_selection import additive_scores, balanced_accuracy

SOURCE = ROOT/'results/nested-crossfit-classification-rounded'
SELECTION = ROOT/'results/forest-penalty-selection'
FOLDER = ROOT/'results/forest-compact-transfer'


def run(name):
    threadpool_limits(1)
    source = SOURCE/(name+'.json'); selection_file = SELECTION/(name+'.json')
    saved = json.loads(source.read_text()); selection = json.loads(selection_file.read_text())
    X, y = load_dataset(name); records = []; selected_records = iter(selection['records'])
    for i, split in enumerate(saved['splits']):
        ri = np.array(split['reference_indices']); raw, labels = X[ri], y[ri]
        for j, inner in enumerate(split['inner']):
            selected = next(selected_records)
            assert selected['outer_reference_indices'] == ri.tolist()
            assert selected['reference_indices'] == inner['reference_indices'] and selected['validation_indices'] == inner['validation_indices']
            a, b = np.array(inner['reference_indices']), np.array(inner['validation_indices'])
            A, B = preprocess_fit(raw[a], raw[b]); yt, truth = labels[a], labels[b]
            stage = inner['by_C']['1.0']['stage']; policy = stage['policies']['0.8']
            support = policy['candidate']['features']; baseline = policy['fit']; fit = None
            key = selected['selection']['selected']
            if key != 'additive':
                if len(support) == A.shape[1]:
                    fit = selected['forest_fit']
                else:
                    fit = fit_forest_logistic(A, yt, stage['teacher_coef'], stage['teacher_intercept'],
                        stage['teacher_classes'], support, selected['tree']['edges'], C=1., pair_penalty=float(key))
            classes = np.array(stage['teacher_classes']); rng = np.random.default_rng(selected['query_seed'])
            queries = [B]+[np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1) for _ in range(3)]
            values = {'additive': [], 'selected': []}
            for query in queries:
                ordinary = additive_scores(query, baseline)
                S = forest_scores(query, fit['coef'], fit['intercept'], fit['edges'],
                    np.asarray(fit['pair_coef']).reshape(len(classes), len(fit['edges']), 3, 3)) if fit is not None and fit['feasible'] else ordinary
                values['additive'].append(balanced_accuracy(ordinary, classes, truth))
                values['selected'].append(balanced_accuracy(S, classes, truth))
            records.append({'outer_reference_indices': ri.tolist(), 'reference_indices': a.tolist(),
                'validation_indices': b.tolist(), 'features': support, 'policy': policy,
                'selected_option': key, 'forest_fit': fit,
                'fallback': fit is not None and not fit['feasible'], 'query_seed': selected['query_seed'],
                'metrics': {m: {'clean': v[0], 'noise': float(np.mean(v[1:]))} for m, v in values.items()},
                'full_metrics': selected['metrics'], 'removed': 1-len(support)/A.shape[1]})
            print(name, i, j, key, len(support), 'gain_pp', 100*(values['selected'][0]-values['additive'][0]), flush=True)
    assert next(selected_records, None) is None
    files = [source, selection_file, Path(__file__), ROOT/'research/forest-compact-transfer-protocol.md',
        ROOT/'src/certified_frfs/centered_forest_fuzzy_logistic_2d.py', ROOT/'data/processed'/(name+'.npz')]
    (FOLDER/(name+'.json')).write_text(json.dumps({'dataset': name, 'records': records,
        'fingerprints': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}, indent=2), encoding='utf-8')


def summarize(names):
    rows = []; counts = Counter()
    for name in names:
        records = json.loads((FOLDER/(name+'.json')).read_text())['records']
        counts['references'] += len(records)
        counts['forest_attempts'] += sum(r['forest_fit'] is not None for r in records)
        counts['forest_failures'] += sum(r['fallback'] for r in records)
        rows.append({'dataset': name, 'removed': float(np.mean([r['removed'] for r in records])),
            **{m: {metric: float(np.mean([r['metrics'][m][metric] for r in records])) for metric in ('clean', 'noise')}
                for m in ('additive', 'selected')},
            'full': {m: {metric: float(np.mean([r['full_metrics'][m][metric] for r in records])) for metric in ('clean', 'noise')}
                for m in ('additive', 'selected')}})
    assert len(rows) == 15 and counts['references'] == 128
    clean = paired([r['selected']['clean']-r['additive']['clean'] for r in rows]); removed = float(np.mean([r['removed'] for r in rows]))
    result = {'rows': rows, 'counts': dict(counts), 'removed': removed, 'clean_gain': clean,
        'noise_gain': paired([r['selected']['noise']-r['additive']['noise'] for r in rows]),
        'vs_selected_full': paired([r['selected']['clean']-r['full']['selected']['clean'] for r in rows]),
        'gate_passed': clean['mean_pp'] >= .5 and removed >= .1 and counts['forest_failures']/max(1, counts['forest_attempts']) <= .1,
        'scope': 'fixed_certified_compact_policy_development_transfer_not_nested_selection_or_SOTA_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({k: v for k, v in result.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    import sys
    available_only = '--available' in sys.argv
    assert available_only or (SELECTION/'complete.json').exists()
    FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    ready = [name for name in names if (SELECTION/(name+'.json')).exists()]
    assert available_only or len(ready) == len(names)
    pending = [name for name in ready if not (FOLDER/(name+'.json')).exists()]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in pending]):
            future.result()
    if all((FOLDER/(name+'.json')).exists() for name in names):
        summarize(names)
    else:
        print('Incomplete: awaiting remaining source datasets; no all15 summary.', flush=True)
