"""Score the entire frozen candidate pool on INNER validation only."""
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from diagnose_repair_bottlenecks_v7 import ROOT, NAMES, digest, ba, blockers

FOLDER = ROOT/'results/rejected-pool-opportunity-v7'


def predict(A, y, B, fs, weights):
    distance = np.zeros((len(B), len(A)))
    for a, w in zip(fs, weights, strict=True):
        distance += w * abs(B[:, a, None] - A[None, :, a])
    nn = np.argsort(distance, axis=1, kind='stable')[:, :min(5, len(A))]
    classes = np.unique(y)
    count = np.column_stack([(y[nn] == c).sum(axis=1) for c in classes])
    return classes[count.argmax(axis=1)]


def run(name):
    path = FOLDER/(name+'.json')
    if path.exists():
        raise FileExistsError(path)
    p = ROOT/'results/repair-relation-selector-v4'/(name+'.json')
    sha = digest(p); source = json.loads(p.read_text()); assert source['complete']
    X, y = load_dataset(name); records, counts = [], Counter()
    for panel in source['rows']:
        ri = np.array(panel['train_indices'], int)
        for fold, block in enumerate(panel['fit']['inner']):
            ii = np.array(block['fit_indices'], int); jj = np.array(block['validation_indices'], int)
            assert set(ii).isdisjoint(jj) and set(ii) | set(jj) == set(range(len(ri)))
            A, B = preprocess_fit(X[ri[ii]], X[ri[jj]])
            labels, truth = y[ri[ii]], y[ri[jj]]
            candidates = {}; cache = {}
            for family, part in block['families'].items():
                for recipe in part['candidates']:
                    for c in recipe['trace']:
                        ref = c['family_certificate']['dependency_certificate'].get('full_weights', c['weights'])
                        candidates[(family, tuple(c['features']), tuple(c['weights']), tuple(ref))] = c
            evaluated = []
            for (family, fs, weights, ref), c in candidates.items():
                key = (fs, weights)
                if key not in cache:
                    pred = predict(A, labels, B, fs, weights)
                    cache[key] = {'predictions': pred.tolist(), 'inner_BA': ba(truth, pred)}
                    counts['distinct_inner_predictions'] += 1
                failure = blockers(c)
                dep = any('dependency' in f for f in failure)
                majority = any('majority' in f for f in failure)
                category = ('both_rejected' if dep and majority else 'dependency_only_rejected'
                            if dep else 'majority_only_rejected' if majority else 'admitted')
                counts[category] += 1
                evaluated.append({'family': family, 'features': list(fs), 'weights': list(weights),
                                  'reference_weights': list(ref), 'blockers': failure,
                                  'category': category, **cache[key]})
            for part in block['families'].values():
                for recipe in part['candidates']:
                    model, score = recipe['model'], recipe['score']
                    key = tuple(model['features']), tuple(model['weights'])
                    assert cache[key]['predictions'] == score['predictions']
                    assert abs(cache[key]['inner_BA'] - score['clean_BA']) < 1e-12
                    counts['saved_prediction_score_replays'] += 1
            def best(vals):
                if not vals:
                    return None
                v = max(vals, key=lambda r: (r['inner_BA'], -len(r['features']), tuple(-a for a in r['features'])))
                return {k: v[k] for k in ('family', 'features', 'inner_BA', 'category', 'blockers')}
            groups = {cat: best([r for r in evaluated if r['category'] == cat])
                      for cat in ('admitted', 'dependency_only_rejected', 'majority_only_rejected', 'both_rejected')}
            rejected = best([r for r in evaluated if r['category'] != 'admitted'])
            allbest = best(evaluated); admitted = groups['admitted']; assert admitted is not None
            records.append({'panel': panel['fold'], 'inner_fold': fold, 'candidates': evaluated,
                'category_best': groups, 'best_rejected': rejected, 'best_any': allbest,
                'rejected_minus_admitted_pp': None if rejected is None else 100*(rejected['inner_BA']-admitted['inner_BA']),
                'any_minus_admitted_pp': 100*(allbest['inner_BA']-admitted['inner_BA'])})
            print(name, panel['fold'], fold, 'rejected/admitted pp', records[-1]['rejected_minus_admitted_pp'], flush=True)
    assert digest(p) == sha
    result = {'dataset': name, 'complete': True, 'records': records, 'counts': dict(counts),
              'source_sha256': sha, 'runner_sha256': digest(__file__),
              'new_outer_scoring': False, 'new_training_fit': False,
              'selected_method_or_certificate_changed': False, 'goal_achieved': False,
              'scope': 'exposed_inner_per_block_oracle_diagnostic_not_deployable_selection_or_risk'}
    path.write_text(json.dumps(result, separators=(',', ':')), encoding='utf8')
    return result


def main():
    threadpool_limits(1); FOLDER.mkdir(exist_ok=True)
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in (
        Path(__file__), ROOT/'scripts/diagnose_repair_bottlenecks_v7.py',
        ROOT/'src/certified_frfs/datasets.py', ROOT/'src/certified_frfs/validated_pool.py',
        ROOT/'data/manifest.json', ROOT/'research/rejected-pool-opportunity-v7-protocol.md')}
    for name in NAMES:
        for p in (ROOT/'results/repair-relation-selector-v4'/(name+'.json'), ROOT/'data/processed'/(name+'.npz')):
            hashes[p.relative_to(ROOT).as_posix()] = digest(p)
    freeze = FOLDER/'source-freeze.json'; assert not freeze.exists()
    freeze.write_text(json.dumps({'hashes': hashes, 'created_unix': time.time(),
        'no_outer_data_access': True}, indent=2), encoding='utf8')
    rows = [run(n) for n in NAMES]
    for rel, value in hashes.items():
        assert digest(ROOT/rel) == value
    summary_rows = []
    for row in rows:
        vals = [r['rejected_minus_admitted_pp'] for r in row['records'] if r['rejected_minus_admitted_pp'] is not None]
        summary_rows.append({'dataset': row['dataset'], 'blocks': len(row['records']),
            'rejected_available_blocks': len(vals),
            'mean_rejected_minus_admitted_pp': float(np.mean(vals)) if vals else None,
            'mean_any_minus_admitted_pp': float(np.mean([r['any_minus_admitted_pp'] for r in row['records']])),
            'rejected_better_blocks': sum(v > 1e-12 for v in vals), 'counts': row['counts']})
    summary = {'complete': True, 'datasets': 6, 'inner_blocks': 54, 'rows': summary_rows,
        'dataset_equal_mean_any_minus_admitted_pp': float(np.mean([r['mean_any_minus_admitted_pp'] for r in summary_rows])),
        'new_outer_scoring': False, 'goal_achieved': False, 'classification_superiority_established': False,
        'artifact_hashes': {n: digest(FOLDER/(n+'.json')) for n in NAMES},
        'limits': ['Per-block validation winners are optimistic oracles, not a tuning recipe.',
                   'Rejected metrics may truly violate original guarantees and cannot be deployed.',
                   'Support/weight differences do not isolate certification as a causal factor.']}
    (FOLDER/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
