"""All15/128inner capacity diagnostic under identical original teacher guards."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from review_box_iterations import ROOT, paired
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.triangular_classifier import triangular_memberships
from certified_frfs.fuzzy_interaction_tree import interaction_tree
from certified_frfs.centered_forest_fuzzy_logistic import fit_forest_logistic
from certified_frfs.forest_fuzzy_classifier import forest_scores

SOURCE = ROOT/'results/nested-crossfit-classification-rounded'
FOLDER = ROOT/'results/centered-fuzzy-forest-capacity'


def BA(scores, classes, truth):
    pred = classes[scores.argmax(axis=1)]
    return float(np.mean([np.mean(pred[truth == c] == c) for c in np.unique(truth)]))


def run(name):
    threadpool_limits(1)
    source = SOURCE/(name+'.json'); saved = json.loads(source.read_text())
    X, y = load_dataset(name); records = []
    for i, split in enumerate(saved['splits']):
        ri = np.array(split['reference_indices']); raw, labels = X[ri], y[ri]
        for j, inner in enumerate(split['inner']):
            a, b = np.array(inner['reference_indices']), np.array(inner['validation_indices'])
            A, B = preprocess_fit(raw[a], raw[b]); yt, truth = labels[a], labels[b]
            stage = inner['by_C']['1.0']['stage']
            baseline = stage['policies']['1.0']['fit']
            tree = interaction_tree(A, yt)
            fit = fit_forest_logistic(A, yt, stage['teacher_coef'], stage['teacher_intercept'],
                stage['teacher_classes'], list(range(A.shape[1])), tree['edges'], C=1.)
            classes = np.array(stage['teacher_classes']); rng = np.random.default_rng(5082026+100*i+j)
            queries = [B]+[np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1) for _ in range(3)]
            values = {'additive': [], 'forest': []}
            for query in queries:
                ordinary = np.einsum('ndk,qdk->nq', triangular_memberships(query), np.array(baseline['coef']))+baseline['intercept']
                scores = forest_scores(query, fit['coef'], fit['intercept'], fit['edges'], fit['pair_coef']) if fit['feasible'] else ordinary
                values['additive'].append(BA(ordinary, classes, truth)); values['forest'].append(BA(scores, classes, truth))
            records.append({'outer_reference_indices': ri.tolist(), 'reference_indices': a.tolist(), 'validation_indices': b.tolist(),
                'teacher_C': 1., 'tree': tree, 'forest_fit': fit, 'full_additive': baseline,
                'fallback': not fit['feasible'], 'query_seed': 5082026+100*i+j,
                'metrics': {method: {'clean': v[0], 'noise': float(np.mean(v[1:]))} for method, v in values.items()}})
            print(name, i, j, fit['status'], 'rounds', len(fit.get('trace', [])), 'forestBA', values['forest'][0], flush=True)
    files = [source, Path(__file__), ROOT/'src/certified_frfs/forest_fuzzy_classifier.py',
             ROOT/'src/certified_frfs/centered_forest_fuzzy_logistic.py', ROOT/'src/certified_frfs/fuzzy_interaction_tree.py',
             ROOT/'research/centered-fuzzy-forest-capacity-protocol.md', ROOT/'data/processed'/(name+'.npz')]
    (FOLDER/(name+'.json')).write_text(json.dumps({'dataset': name, 'records': records,
        'fingerprints': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            future.result()
    rows, counts = [], {'inner_fits': 0, 'failures': 0}
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); records = saved['records']
        counts['inner_fits'] += len(records); counts['failures'] += sum(r['fallback'] for r in records)
        rows.append({'dataset': name, **{method: {metric: float(np.mean([r['metrics'][method][metric] for r in records]))
                    for metric in ('clean', 'noise')} for method in ('additive', 'forest')},
                    'failures': sum(r['fallback'] for r in records)})
    assert counts['inner_fits'] == 128 and len(rows) == 15
    clean = paired([r['forest']['clean']-r['additive']['clean'] for r in rows])
    result = {'rows': rows, 'counts': counts, 'clean_gain': clean,
        'noise_gain': paired([r['forest']['noise']-r['additive']['noise'] for r in rows]),
        'gate_passed': clean['mean_pp'] >= .5 and counts['failures']/128 <= .1,
        'scope': 'all15_all128inner_full_attribute_capacity_diagnostic_not_feature_selection_or_outer_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({key: value for key, value in result.items() if key != 'rows'}, flush=True)
