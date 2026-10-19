"""All 128 references: fresh reference-internal pair penalty selection."""
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
from certified_frfs.fuzzy_interaction_tree import interaction_tree
from certified_frfs.centered_forest_fuzzy_logistic_2d import fit_forest_logistic
from certified_frfs.forest_fuzzy_classifier import forest_scores
from certified_frfs.forest_penalty_selection import select_pair_penalty, additive_scores, balanced_accuracy, OPTIONS

SOURCE = ROOT/'results/nested-crossfit-classification-rounded'
FOLDER = ROOT/'results/forest-penalty-selection'


def run(name):
    threadpool_limits(1)
    source = SOURCE/(name+'.json'); saved = json.loads(source.read_text())
    X, y = load_dataset(name); records = []
    for i, split in enumerate(saved['splits']):
        ri = np.array(split['reference_indices']); raw, labels = X[ri], y[ri]
        for j, inner in enumerate(split['inner']):
            a, b = np.array(inner['reference_indices']), np.array(inner['validation_indices'])
            selection = select_pair_penalty(raw[a], labels[a]); key = selection['selected']
            A, B = preprocess_fit(raw[a], raw[b]); yt, truth = labels[a], labels[b]
            stage = inner['by_C']['1.0']['stage']; baseline = stage['policies']['1.0']['fit']
            tree = interaction_tree(A, yt); fit = None
            if key != 'additive':
                fit = fit_forest_logistic(A, yt, stage['teacher_coef'], stage['teacher_intercept'],
                    stage['teacher_classes'], list(range(A.shape[1])), tree['edges'], C=1., pair_penalty=float(key))
            classes = np.array(stage['teacher_classes']); rng = np.random.default_rng(5082026+100*i+j)
            queries = [B]+[np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1) for _ in range(3)]
            values = {'additive': [], 'selected': []}
            for query in queries:
                ordinary = additive_scores(query, baseline)
                scores = forest_scores(query, fit['coef'], fit['intercept'], fit['edges'], fit['pair_coef']) if fit is not None and fit['feasible'] else ordinary
                values['additive'].append(balanced_accuracy(ordinary, classes, truth))
                values['selected'].append(balanced_accuracy(scores, classes, truth))
            records.append({'outer_reference_indices': ri.tolist(), 'reference_indices': a.tolist(),
                'validation_indices': b.tolist(), 'selection': selection, 'tree': tree,
                'forest_fit': fit, 'full_additive': baseline,
                'fallback': fit is not None and not fit['feasible'], 'query_seed': 5082026+100*i+j,
                'metrics': {m: {'clean': v[0], 'noise': float(np.mean(v[1:]))} for m, v in values.items()}})
            print(name, i, j, key, 'gain_pp', 100*(values['selected'][0]-values['additive'][0]), flush=True)
    files = [source, Path(__file__), ROOT/'research/forest-penalty-selection-protocol.md', ROOT/'data/processed'/(name+'.npz')]
    files += [ROOT/'src/certified_frfs'/f for f in ('forest_penalty_selection.py', 'centered_forest_fuzzy_logistic_2d.py',
        'forest_fuzzy_classifier.py', 'fuzzy_interaction_tree.py', 'nested_teacher_regularized.py',
        'validated_pool.py', 'linear_retention.py', 'triangular_classifier.py', 'constrained_fuzzy_logistic.py')]
    (FOLDER/(name+'.json')).write_text(json.dumps({'dataset': name, 'records': records,
        'fingerprints': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}, indent=2), encoding='utf-8')


def summarize(names):
    rows = []; counts = Counter(); selections = Counter()
    for name in names:
        records = json.loads((FOLDER/(name+'.json')).read_text())['records']
        for r in records:
            counts['references'] += 1; selections[r['selection']['selected']] += 1
            counts['final_forest_attempts'] += r['forest_fit'] is not None
            counts['final_forest_failures'] += r['fallback']
            for fold in r['selection']['folds']:
                counts['selection_folds'] += 1
                counts['selection_forest_fits'] += len(fold['forest_fits'])
                counts['selection_forest_failures'] += sum(not f['feasible'] for f in fold['forest_fits'].values())
                counts['selection_additive_failures'] += fold['additive_failure'] is not None
        rows.append({'dataset': name, **{m: {metric: float(np.mean([r['metrics'][m][metric] for r in records]))
            for metric in ('clean', 'noise')} for m in ('additive', 'selected')}})
    assert len(rows) == 15 and counts['references'] == 128
    clean = paired([r['selected']['clean']-r['additive']['clean'] for r in rows])
    result = {'rows': rows, 'counts': dict(counts), 'selection_counts': dict(selections), 'clean_gain': clean,
        'noise_gain': paired([r['selected']['noise']-r['additive']['noise'] for r in rows]),
        'gate_passed': clean['mean_pp'] >= .5 and counts['final_forest_failures']/max(1, counts['final_forest_attempts']) <= .1,
        'scope': 'full_support_reference_internal_selection_development_capacity_not_FS_or_outer_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({k: v for k, v in result.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            future.result()
    summarize(names)
