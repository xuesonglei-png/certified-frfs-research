"""Interaction-aware support path; no query argument in the path generator."""
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
from certified_frfs.forest_certified_backward import forest_certified_backward_path
from certified_frfs.forest_fuzzy_classifier import forest_scores
from certified_frfs.forest_penalty_selection import additive_scores, balanced_accuracy

SOURCE = ROOT/'results/nested-crossfit-classification-rounded'
SELECTION = ROOT/'results/forest-penalty-selection'
COMPACT = ROOT/'results/forest-compact-transfer'
FOLDER = ROOT/'results/forest-aware-support'


def run(name):
    threadpool_limits(1)
    sources = [SOURCE/(name+'.json'), SELECTION/(name+'.json'), COMPACT/(name+'.json')]
    saved, selected, compact = [json.loads(p.read_text()) for p in sources]
    X, y = load_dataset(name); records = []; counter = 0
    for i, split in enumerate(saved['splits']):
        ri = np.array(split['reference_indices']); raw, labels = X[ri], y[ri]
        for j, inner in enumerate(split['inner']):
            full, old = selected['records'][counter], compact['records'][counter]; counter += 1
            for r in (full, old):
                assert r['outer_reference_indices'] == ri.tolist()
                assert r['reference_indices'] == inner['reference_indices'] and r['validation_indices'] == inner['validation_indices']
            a, b = np.array(inner['reference_indices']), np.array(inner['validation_indices'])
            A, B = preprocess_fit(raw[a], raw[b]); yt, truth = labels[a], labels[b]
            stage = inner['by_C']['1.0']['stage']; classes = np.array(stage['teacher_classes'])
            key = full['selection']['selected']; path = None; fs = old['features']; fit = old['forest_fit']
            if key != 'additive':
                path = forest_certified_backward_path(A, yt, classes, full['forest_fit'],
                    stage['teacher_coef'], stage['teacher_intercept'], pair_penalty=float(key), target_size=len(fs))
                final = path['trace'][-1]; fs, fit = final['features'], final['fit']
            rng = np.random.default_rng(old['query_seed'])
            queries = [B]+[np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1) for _ in range(3)]
            values = []
            for query in queries:
                S = forest_scores(query, fit['coef'], fit['intercept'], fit['edges'],
                    np.asarray(fit['pair_coef']).reshape(len(classes), len(fit['edges']), 3, 3)) if key != 'additive' else additive_scores(query, old['policy']['fit'])
                values.append(balanced_accuracy(S, classes, truth))
            records.append({'outer_reference_indices': ri.tolist(), 'reference_indices': a.tolist(),
                'validation_indices': b.tolist(), 'selected_option': key, 'path': path,
                'features': fs, 'fit': fit, 'target_features': old['features'],
                'budget_matched': len(fs) == len(old['features']), 'removed': 1-len(fs)/A.shape[1],
                'query_seed': old['query_seed'], 'metrics': {'clean': values[0], 'noise': float(np.mean(values[1:]))},
                'old_metrics': old['metrics']['selected'], 'full_metrics': full['metrics']['selected'], 'old_removed': old['removed']})
            print(name, i, j, key, len(fs), '/', len(old['features']), 'gain_pp', 100*(values[0]-old['metrics']['selected']['clean']), flush=True)
    assert counter == len(selected['records']) == len(compact['records'])
    files = sources+[Path(__file__), ROOT/'research/forest-aware-support-protocol.md', ROOT/'data/processed'/(name+'.npz')]
    files += [ROOT/'src/certified_frfs'/f for f in ('forest_certified_backward.py', 'forest_projection.py',
        'centered_forest_fuzzy_logistic_batched.py', 'forest_fuzzy_classifier.py', 'feasible_transport.py',
        'redistributed_loss.py', 'capped_sparse.py', 'majority_metric.py', 'validated_pool.py')]
    (FOLDER/(name+'.json')).write_text(json.dumps({'dataset': name, 'records': records,
        'fingerprints': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}, indent=2), encoding='utf-8')


def summarize(names):
    rows = []; counts = Counter()
    for name in names:
        records = json.loads((FOLDER/(name+'.json')).read_text())['records']
        for r in records:
            counts['references'] += 1; counts['budget_unmatched'] += not r['budget_matched']
            if r['path'] is not None:
                counts['forest_paths'] += 1
                counts['attempts'] += len(r['path']['attempts'])
                counts['failed_attempts'] += sum(not a['accepted'] for a in r['path']['attempts'])
                counts['failed_classifier_fits'] += sum(a['fit'] is not None and not a['fit']['feasible'] for a in r['path']['attempts'])
        rows.append({'dataset': name, 'removed': float(np.mean([r['removed'] for r in records])),
            'old_removed': float(np.mean([r['old_removed'] for r in records])),
            **{m: {metric: float(np.mean([r[field][metric] for r in records])) for metric in ('clean', 'noise')}
                for m, field in (('new', 'metrics'), ('old', 'old_metrics'), ('full', 'full_metrics'))}})
    assert len(rows) == 15 and counts['references'] == 128
    gain = paired([r['new']['clean']-r['old']['clean'] for r in rows])
    removed = float(np.mean([r['removed'] for r in rows])); old_removed = float(np.mean([r['old_removed'] for r in rows]))
    result = {'rows': rows, 'counts': dict(counts), 'removed': removed, 'old_removed': old_removed,
        'clean_gain': gain, 'noise_gain': paired([r['new']['noise']-r['old']['noise'] for r in rows]),
        'vs_full': paired([r['new']['clean']-r['full']['clean'] for r in rows]),
        'gate_passed': gain['mean_pp'] >= .5 and removed >= old_removed-1e-12,
        'scope': 'development_interaction_aware_support_mechanism_not_nested_method_or_SOTA_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({k: v for k, v in result.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    assert (COMPACT/'complete-audit.json').exists() and (SELECTION/'complete-audit.json').exists()
    FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]): future.result()
    summarize(names)
