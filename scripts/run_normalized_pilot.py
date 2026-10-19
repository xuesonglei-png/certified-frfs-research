"""Separate exploratory model: normalize each subset's distance by its weight.

This is a different preservation target from either first-cycle absolute loss
or fixed-additive relative retention. It is NOT a superior solver for the same
problem. Settings are fixed here before running this first-split pilot.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from threadpoolctl import threadpool_limits
from certified_frfs.core import positive_region
from certified_frfs.datasets import available_datasets, load_dataset
from certified_frfs.selection import deletion_order
from certified_frfs.normalized_relative import normalized_relative_certificate, select_normalized_relative

CONFIGS = [
    {'id': 'normalized_class_q90_e01_independent', 'eps': .01, 'method': 'independent'},
    {'id': 'normalized_class_q90_e01_hybrid', 'eps': .01, 'method': 'hybrid'},
    {'id': 'normalized_class_q90_e00_independent', 'eps': 0., 'method': 'independent'},
]
OUTPUT = ROOT/'results/round2/normalized-pilot.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    return {str(p.relative_to(ROOT)): digest(p)
            for p in [Path(__file__), *sorted((ROOT/'src/certified_frfs').glob('*.py'))]}


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def metrics(model, Xte, yte, labels, features):
    pred = model.predict(Xte[:, features])
    return {'balanced_accuracy': float(balanced_accuracy_score(yte, pred)),
            'macro_f1': float(f1_score(yte, pred, average='macro', zero_division=0)),
            'per_class_recall': recall_score(yte, pred, labels=labels, average=None, zero_division=0).tolist()}


def run_dataset(name):
    path = ROOT/'results/primary'/f'{name}-s17-f0.json'
    previous = json.loads(path.read_text(encoding='utf-8'))
    X, y = load_dataset(name)
    train, test = np.array(previous['train_indices']), np.array(previous['test_indices'])
    assert not set(train) & set(test)
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    scaler = MinMaxScaler(clip=True)
    Xtr = scaler.fit_transform(imputer.fit_transform(X[train]))
    Xte = scaler.transform(imputer.transform(X[test]))
    assert np.allclose(scaler.data_min_, previous['scaler_min'], rtol=0, atol=1e-14)
    assert np.allclose(scaler.data_max_, previous['scaler_max'], rtol=0, atol=1e-14)
    assert np.allclose(imputer.statistics_, previous['imputer_statistics'], rtol=0, atol=1e-14)
    ytr, yte = y[train], y[test]
    d, labels = Xtr.shape[1], np.unique(ytr)
    weights = np.array(previous['weights'])
    order = deletion_order(Xtr, ytr, seed=17)
    attempts = [s['removed_candidate'] for s in previous['deletion_history']]
    assert attempts == [a for a in order if a in attempts]
    selections = []
    for config in CONFIGS:
        start = perf_counter()
        selected = select_normalized_relative(Xtr, ytr, weights=weights, order=order,
                     retention=.9, scope='classwise', eps=config['eps'], method=config['method'])
        assert selected['certificate']['certified']
        selected.update(config, n_features=len(selected['features']),
                        removed_fraction=1-len(selected['features'])/d, selection_seconds=perf_counter()-start)
        selections.append(selected)
        print(f"{name}/{config['id']}: {selected['n_features']}/{d}", flush=True)
    picked = selections[1]['features']
    matched = sorted(order[-len(picked):])
    predictors = {'all': list(range(d)), 'normalized_hybrid': picked,
                  'mi_matched': matched, 'normalized_nominal': selections[2]['features']}
    models = {key: KNeighborsClassifier(n_neighbors=5).fit(Xtr[:, fs], ytr)
              for key, fs in predictors.items()}
    evaluations = []
    base_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')
    rng = np.random.default_rng(base_seed+17)
    for noise in [0., .01, .05]:
        for repetition in range(1 if noise == 0 else 5):
            Z = Xte if noise == 0 else np.clip(Xte+rng.uniform(-noise, noise, Xte.shape), 0., 1.)
            for key, fs in predictors.items():
                evaluations.append({'method': key, 'test_epsilon': noise, 'repetition': repetition,
                                    **metrics(models[key], Z, yte, labels, fs)})
    clean_all = next(r for r in evaluations if r['method'] == 'all' and r['test_epsilon'] == 0)
    old_all = next(r for r in previous['rows'] if r['method'] == 'all' and r['classifier'] == 'knn' and r['test_epsilon'] == 0)
    assert np.isclose(clean_all['balanced_accuracy'], old_all['balanced_accuracy'], atol=1e-14, rtol=0)
    fixed_certificate = json.loads((ROOT/'results/round2/relative-pilot.json').read_text(encoding='utf-8'))
    fixed_row = next(r for r in fixed_certificate['rows'] if r['dataset'] == name)
    return {'dataset': name, 'd': d, 'n_train': len(train), 'n_test': len(test), 'seed': 17, 'fold': 0,
            'train_indices': train, 'test_indices': test, 'labels': labels, 'weights': weights,
            'source_run_sha256': digest(path), 'processed_data_sha256': digest(ROOT/'data/processed'/f'{name}.npz'),
            'mi_deletion_order': order, 'configurations': selections,
            'matched_mi_features': matched, 'normalized_equals_mi_matched': sorted(picked) == matched,
            'hybrid_additional_deleted': selections[0]['n_features']-selections[1]['n_features'],
            'hybrid_changed_subset': selections[0]['features'] != selections[1]['features'],
            'evaluations': evaluations,
            'first_stage_fixed_relative_features': fixed_row['configurations'][1]['features']}


def summarize(rows):
    summary = {'datasets': len(rows), 'configurations': []}
    for config in CONFIGS:
        chosen = [next(s for s in row['configurations'] if s['id'] == config['id']) for row in rows]
        summary['configurations'].append({**config,
            'mean_removed_fraction': float(np.mean([s['removed_fraction'] for s in chosen])),
            'zero_deletion_datasets': [row['dataset'] for row, sel in zip(rows, chosen) if sel['n_features'] == row['d']]})
    summary['predictive'] = []
    for method in ['all', 'normalized_hybrid', 'mi_matched', 'normalized_nominal']:
        for noise in [0., .01, .05]:
            per_dataset = [np.mean([r['balanced_accuracy'] for r in row['evaluations']
                                    if r['method'] == method and r['test_epsilon'] == noise]) for row in rows]
            summary['predictive'].append({'method': method, 'test_epsilon': noise,
                    'mean_balanced_accuracy': float(np.mean(per_dataset))})
    summary['hybrid_additional_deleted_total'] = sum(r['hybrid_additional_deleted'] for r in rows)
    summary['hybrid_changed_subset_datasets'] = [r['dataset'] for r in rows if r['hybrid_changed_subset']]
    summary['identical_matched_mi_subsets'] = sum(r['normalized_equals_mi_matched'] for r in rows)
    return summary


def main():
    started = perf_counter()
    initial = sources()
    result = {'protocol': 'normalized-relative-first-split-2026-09-20',
        'analysis_status': 'post_hoc_new_model_exploration_not_replacement_for_primary',
        'model': 'Each nonempty subset normalizes its weighted L1 distance by its own positive weight sum; domain [0,1]',
        'caveat': 'Different information measure from fixed-additive model; no attribute monotonicity or predictive guarantee',
        'configurations_fixed_before_run': CONFIGS, 'retention': .9, 'scope': 'classwise',
        'stability': 'interval', 'classifier': '5NN Euclidean, fitted only on original training split',
        'test_noise_repetitions': 5, 'test_noise': 'shared uniform coordinate perturbations for all methods, clipped to [0,1]',
        'started_utc': datetime.now(timezone.utc).isoformat(), 'source_hashes': initial,
        'data_manifest_sha256': digest(ROOT/'data/manifest.json'), 'rows': []}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        with threadpool_limits(limits=1):
            for name in available_datasets():
                result['rows'].append(run_dataset(name))
                result['run_status'] = 'running'
                OUTPUT.write_text(json.dumps(result, indent=2, default=plain, allow_nan=False)+'\n', encoding='utf-8')
        result['source_hashes_end'] = sources()
        assert initial == result['source_hashes_end'], 'Source code changed during the pilot'
    except BaseException as error:
        result.update(run_status='failed', error=f'{type(error).__name__}: {error}',
                      seconds=perf_counter()-started, source_hashes_end=sources())
        OUTPUT.write_text(json.dumps(result, indent=2, default=plain, allow_nan=False)+'\n', encoding='utf-8')
        raise
    result.update(run_status='complete', summary=summarize(result['rows']),
                  seconds=perf_counter()-started, source_files_unchanged_during_run=True,
                  completed_utc=datetime.now(timezone.utc).isoformat())
    OUTPUT.write_text(json.dumps(result, indent=2, default=plain, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
