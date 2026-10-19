"""All-six inner-only diagnosis; matched supports and certification are distinct.

No outer scores, labels, predictions, or test indices are consulted. This is a
mechanism diagnosis on exposed development data, not a new generalization study.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from certified_frfs.datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('banknote', 'glass', 'heart_statlog', 'ionosphere', 'seeds', 'wdbc')
OUT = ROOT / 'research/repair-bottlenecks-inner-v7.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ba(truth, pred):
    truth, pred = np.asarray(truth), np.asarray(pred)
    assert pred.shape == truth.shape
    return float(np.mean([np.mean(pred[truth == c] == c) for c in np.unique(truth)]))


def selected(options, blocks, method):
    choices = []
    for i in range(len(options[method])):
        records = [b['options'][method][i] for b in blocks]
        assert len({r['config_index'] for r in records}) == 1
        if any(r['score'] is None for r in records):
            continue
        choices.append({'index': i, 'config_index': records[0]['config_index'],
                        'BA': float(np.mean([r['score']['clean_BA'] for r in records])),
                        'logloss': float(np.mean([r['score']['balanced_logloss'] for r in records])),
                        'count': float(np.mean([len(r['metric']['features']) for r in records]))})
    if not choices:
        return None
    return max(choices, key=lambda x: (x['BA'], -x['logloss'], -x['count'], -x['index']))


def blockers(candidate):
    failures = []
    for side, cert_key, class_key in (
            ('original', 'certificate', 'classwise'),
            ('learned', 'family_certificate', 'family_classwise')):
        cert = candidate[cert_key]
        dependency_ok = cert['dependency_certificate']['upper_bound'] <= .1
        assert bool(cert['feasible']) == bool(dependency_ok and cert['majority_feasible'])
        if not dependency_ok:
            failures.append(side + '_dependency')
        if not candidate[class_key]['feasible']:
            failures.append(side + '_classwise_dependency')
        if not cert['majority_feasible']:
            failures.append(side + '_majority')
    assert bool(candidate['joint_feasible']) == (not failures)
    return failures


def main():
    if OUT.exists():
        raise FileExistsError('Versioned diagnostic output already exists: ' + str(OUT))
    rows, source_hashes, guard_rows = [], {}, []
    replay_count = 0
    replay_cache = {}
    for name in NAMES:
        pp = ROOT / 'results/repair-relation-selector-v4' / (name + '.json')
        cp = ROOT / 'results/repair-relation-controls-v4' / (name + '.json')
        source_hashes[pp.relative_to(ROOT).as_posix()] = digest(pp)
        source_hashes[cp.relative_to(ROOT).as_posix()] = digest(cp)
        parent, controls = json.loads(pp.read_text()), json.loads(cp.read_text())
        assert parent['complete'] and controls['complete']
        X, y = load_dataset(name)
        dataset_panels = []
        family_counts = {f: Counter() for f in parent['rows'][0]['fit']['inner'][0]['families']}
        for pr, cr in zip(parent['rows'], controls['rows'], strict=True):
            assert pr['fold'] == cr['fold'] and pr['train_indices'] == cr['train_indices']
            ri = np.asarray(pr['train_indices'], int)
            blocks = cr['inner']
            for pb, cb in zip(pr['fit']['inner'], blocks, strict=True):
                ii, jj = np.asarray(pb['fit_indices'], int), np.asarray(pb['validation_indices'], int)
                assert ii.tolist() == cb['fit_indices'] and jj.tolist() == cb['validation_indices']
                assert set(ii).isdisjoint(jj) and set(ii) | set(jj) == set(range(len(ri)))
                truth = y[ri[jj]]
                # Every distinct saved inner prediction is replayed against
                # training-panel validation labels, including every baseline grid.
                scores = [o['score'] for arr in cb['options'].values() for o in arr if o['score'] is not None]
                scores += [r['score'] for part in pb['families'].values() for r in part['candidates']]
                for score in scores:
                    key = (name, tuple(ri[jj]), tuple(score['predictions']))
                    if key not in replay_cache:
                        replay_cache[key] = ba(truth, score['predictions'])
                        replay_count += 1
                    assert abs(replay_cache[key] - score['clean_BA']) < 1e-12
                for family, part in pb['families'].items():
                    counts = family_counts[family]
                    unique = {}
                    for recipe in part['candidates']:
                        model = recipe['model']
                        counts['recipes'] += 1
                        counts['fallbacks'] += int(model['uniform_fallback'])
                        counts['full_support_recipes'] += int(len(model['features']) == X.shape[1])
                        counts['restored_features'] += model['restored_count']
                        for c in recipe['trace']:
                            unique[(tuple(c['features']), tuple(c['weights']),
                                    tuple(c['family_certificate']['dependency_certificate'].get('full_weights', c['weights'])))] = c
                    for c in unique.values():
                        failures = blockers(c)
                        counts['distinct_candidates'] += 1
                        counts['feasible_candidates'] += int(not failures)
                        counts['compressed_feasible_candidates'] += int(not failures and len(c['features']) < X.shape[1])
                        counts['any_dependency_block'] += int(any('dependency' in f for f in failures))
                        counts['any_majority_block'] += int(any('majority' in f for f in failures))
                        counts['majority_only_block'] += int(bool(failures) and all('majority' in f for f in failures))
                        for f in failures:
                            counts[f] += 1
                        guard_rows.append({'dataset': name, 'panel': pr['fold'], 'family': family,
                                           'inner_fit_indices': ii.tolist(), 'features': c['features'],
                                           'blockers': failures})
            method_choices = {m: selected(blocks[0]['options'], blocks, m) for m in blocks[0]['options']}
            pool_best = max(pr['fit']['options'], key=lambda x: x['mean_BA'])
            anchor = float(np.mean([b['families']['uniform_sqrt']['candidates'][3]['score']['clean_BA']
                                    for b in pr['fit']['inner']]))
            contrasts = {}
            for algorithm in ('fgs', 'sfrip'):
                for weighting in ('uniform', 'capped_transfer', 'local_soft'):
                    native = algorithm + '_native_' + weighting
                    matched = algorithm + '_matched_' + weighting
                    nc, mc = method_choices[native], method_choices[matched]
                    if nc is None or mc is None:
                        contrasts[native] = {'available': False}
                        continue
                    # Same native-selected configuration removes retuning as a
                    # confounder; separately tuned matched arm is also reported.
                    vals = [b['options'][matched][nc['index']] for b in blocks]
                    assert all(v['config_index'] == nc['config_index'] for v in vals)
                    assert all(v['score'] is not None for v in vals)
                    same_ba = float(np.mean([v['score']['clean_BA'] for v in vals]))
                    same_count = float(np.mean([len(v['metric']['features']) for v in vals]))
                    contrasts[native] = {'available': True,
                        'best_existing_pool_minus_native_pp': 100 * (pool_best['mean_BA'] - nc['BA']),
                        'matched_minus_native_separately_tuned_pp': 100 * (mc['BA'] - nc['BA']),
                        'matched_minus_native_same_config_pp': 100 * (same_ba - nc['BA']),
                        'matched_added_features_same_config': same_count - nc['count'],
                        'native_minimum_cap_count': int(np.ceil(X.shape[1] / 2)),
                        'native_folds_below_cap_count': sum(len(b['options'][native][nc['index']]['metric']['features'])
                                                           < int(np.ceil(X.shape[1] / 2)) for b in blocks)}
            dataset_panels.append({'panel': pr['fold'], 'best_existing_pool': pool_best,
                                   'pool_minus_full_pp': 100 * (pool_best['mean_BA'] - anchor),
                                   'controls': method_choices, 'contrasts': contrasts})
        rows.append({'dataset': name, 'd': X.shape[1], 'panels': dataset_panels,
                     'family_training_counts': {f: dict(c) for f, c in family_counts.items()}})
    summaries = {}
    methods = rows[0]['panels'][0]['contrasts']
    for method in methods:
        ds = []
        for row in rows:
            cc = [p['contrasts'][method] for p in row['panels']]
            if all(c['available'] for c in cc):
                ds.append({'dataset': row['dataset'], **{k: float(np.mean([c[k] for c in cc]))
                           for k in ('best_existing_pool_minus_native_pp',
                                     'matched_minus_native_separately_tuned_pp',
                                     'matched_minus_native_same_config_pp',
                                     'matched_added_features_same_config', 'native_folds_below_cap_count')}})
        summaries[method] = {'available_datasets': len(ds), 'dataset_rows': ds,
            'dataset_equal_means': {k: float(np.mean([r[k] for r in ds])) for k in ds[0] if k != 'dataset'} if ds else {}}
    for rel, sha in source_hashes.items():
        assert digest(ROOT / rel) == sha
    result = {'complete': True, 'datasets': len(rows), 'training_panels': 18, 'inner_blocks': 54,
              'independent_distinct_inner_BA_replays': replay_count,
              'rows': rows, 'contrasts': summaries, 'unique_guard_records': guard_rows,
              'source_hashes': source_hashes, 'script_sha256': digest(__file__),
              'outer_test_indices_labels_or_scores_consulted': False,
              'new_fitting_or_outer_scoring': False, 'classification_superiority_established': False,
              'goal_achieved': False,
              'scope': 'exposed_inner_mechanism_diagnosis_selected_scores_are_optimistic',
              'limits': ['Matched/native support comparisons describe changing support size; they do not isolate guard causality.',
                         'The same-configuration arm is separate from the independently tuned matched control.',
                         'Counts overlap across blockers; distinct candidates are unique within each fold and relation family.',
                         'BA replay does not validate optimizer solutions, certificates, macro-F1, or logloss.',
                         'No independent significance, generalization advantage or publication readiness established.']}
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('complete', 'datasets', 'training_panels', 'inner_blocks',
                                            'independent_distinct_inner_BA_replays')}, indent=2))
    for method, report in summaries.items():
        print(method, report['available_datasets'], report['dataset_equal_means'])


if __name__ == '__main__':
    main()
