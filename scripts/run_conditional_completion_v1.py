"""Frozen six-dataset mechanism pilot. No outer-test tuning, no result overwrite."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from certified_frfs.datasets import load_dataset
from certified_frfs.baselines import mi_scores
from certified_frfs.conditional_completion import ConditionalCompletion1NN

DATASETS = ['glass', 'heart_statlog', 'ionosphere', 'sonar', 'wdbc', 'wine']
DEST = ROOT / 'results' / 'conditional-completion-v1'


def digest():
    files = [Path(__file__), ROOT / 'src/certified_frfs/conditional_completion.py',
             ROOT / 'src/certified_frfs/baselines.py', ROOT / 'src/certified_frfs/datasets.py',
             ROOT / 'research/conditional-completion-v1-protocol.md']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def scaled(train, test):
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    A = imputer.fit_transform(train)
    scaler = MinMaxScaler(clip=True).fit(A)
    return scaler.transform(A), scaler.transform(imputer.transform(test))


def metrics(y, prediction):
    return {'BA': float(balanced_accuracy_score(y, prediction)),
            'macro_F1': float(f1_score(y, prediction, average='macro', zero_division=0))}


def static_select(X, y):
    fractions = [.2, .35, .5, .65, .8, 1.]
    records = [[] for _ in fractions]
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=9192).split(X, y):
        A, B = scaled(X[tr], X[va])
        rank = np.argsort(-mi_scores(A, y[tr], random_state=9192), kind='stable')
        for j, fraction in enumerate(fractions):
            fs = rank[:max(1, int(np.ceil(fraction * A.shape[1])))]
            predictions = y[tr][cdist(B[:, fs], A[:, fs], 'cityblock').argmin(axis=1)]
            records[j].append(metrics(y[va], predictions)['BA'])
    scores = np.mean(records, axis=1)
    chosen = int(np.argmax(scores))
    return fractions[chosen], [{'fraction': f, 'fold_BA': v, 'mean_BA': float(s)}
                               for f, v, s in zip(fractions, records, scores)]


def run(job):
    threadpool_limits(1)
    name, fold = job
    path = DEST / f'{name}-s9191-f{fold}.json'
    hashes = digest()
    if path.exists():
        existing = json.loads(path.read_text(encoding='utf-8'))
        if existing['source_hashes'] != hashes:
            raise RuntimeError('source changed; use a new experiment version')
        return {'dataset': name, 'fold': fold, 'cached': True}
    X, y = load_dataset(name)
    tr, te = list(StratifiedKFold(3, shuffle=True, random_state=9191).split(X, y))[fold]
    start = time.perf_counter()
    A, B = scaled(X[tr], X[te])
    rank = np.argsort(-mi_scores(A, y[tr], random_state=9192), kind='stable').tolist()
    prepare = time.perf_counter() - start
    baseline = ConditionalCompletion1NN(A, y[tr], rank, bound='independent')
    # One warmup and five repetitions, identical scalar per-query Python API.
    baseline.predict_full(B[0])
    times = []
    for _ in range(5):
        start = time.perf_counter()
        teacher = np.asarray([baseline.predict_full(q) for q in B])
        times.append(time.perf_counter() - start)
    teacher_time = float(np.median(times))
    # Independent distance implementation and first-reference-row tie check.
    independent_teacher = y[tr][cdist(B / A.shape[1], A / A.shape[1], 'cityblock').argmin(axis=1)]
    if not np.array_equal(teacher, independent_teacher):
        raise AssertionError('independent full-teacher prediction mismatch')
    rows = {'full': {**metrics(y[te], teacher), 'mean_acquired_fraction': 1.,
                     'online_seconds': teacher_time, 'predictions': teacher.tolist(),
                     'timing_repetitions_seconds': times}}
    for method, bound, adaptive in [('shared_mi', 'shared', False),
                                    ('shared_adaptive', 'shared', True),
                                    ('independent_mi', 'independent', False)]:
        model = ConditionalCompletion1NN(A, y[tr], rank, bound=bound, adaptive=adaptive)
        results = []
        start = time.perf_counter()
        for q in B:
            accessed = []
            def acquire(a):
                if a in accessed:
                    raise AssertionError('repeated query-coordinate access')
                accessed.append(a)
                return q[a]
            result = model.predict_one(acquire)
            if accessed != result['acquired']:
                raise AssertionError('unreported query access')
            results.append(result)
        first_elapsed = time.perf_counter() - start
        # Repeat twice without logging; ensure repeats preserve all predictions.
        online_times = [first_elapsed]
        for _ in range(2):
            start = time.perf_counter()
            repeated = [model.predict_one(lambda a, q=q: q[a])['prediction'] for q in B]
            online_times.append(time.perf_counter() - start)
            if repeated != [r['prediction'] for r in results]:
                raise AssertionError('repeat mismatch')
        prediction = np.asarray([r['prediction'] for r in results])
        if not np.array_equal(prediction, independent_teacher):
            raise AssertionError('conditional prediction disagrees with full teacher')
        counts = np.asarray([r['count'] for r in results])
        elapsed = float(np.median(online_times))
        rows[method] = {**metrics(y[te], prediction), 'teacher_agreement': 1.,
                        'mean_acquired_fraction': float(counts.mean() / A.shape[1]),
                        'p50_acquired_fraction': float(np.median(counts) / A.shape[1]),
                        'p90_acquired_fraction': float(np.quantile(counts, .9) / A.shape[1]),
                        'early_stop_fraction': float(np.mean([r['early_certified'] for r in results])),
                        'online_seconds': elapsed, 'full_over_conditional_speedup': teacher_time / elapsed,
                        'online_repetitions_seconds': online_times,
                        'policy_setup_seconds': model.setup_seconds, 'cache_bytes': model.cache_bytes,
                        'query_records': results}
    start = time.perf_counter()
    fraction, candidates = static_select(X[tr], y[tr])
    fs = rank[:max(1, int(np.ceil(fraction * A.shape[1])))]
    selection_time = time.perf_counter() - start
    start = time.perf_counter()
    static_prediction = y[tr][cdist(B[:, fs], A[:, fs], 'cityblock').argmin(axis=1)]
    static_elapsed = time.perf_counter() - start
    rows['static_mi'] = {**metrics(y[te], static_prediction), 'features': fs,
                         'mean_acquired_fraction': len(fs) / A.shape[1],
                         'teacher_agreement': float(np.mean(static_prediction == teacher)),
                         'inner_candidates': candidates, 'selected_fraction': fraction,
                         'selection_seconds': selection_time, 'batch_online_seconds': static_elapsed,
                         'predictions': static_prediction.tolist()}
    # Static batch timing is explicitly not compared to scalar conditional timing.
    payload = {'dataset': name, 'fold': fold, 'seed': 9191, 'dimension': A.shape[1],
               'train_indices': tr.tolist(), 'test_indices': te.tolist(),
               'test_labels': y[te].tolist(), 'ranking': rank,
               'training_preprocessing_ranking_seconds': prepare,
               'source_hashes': hashes, 'data_sha256': hashlib.sha256(
                   (ROOT / 'data/processed' / f'{name}.npz').read_bytes()).hexdigest(),
               'audit': 'all callback reads logged; scipy full teacher and all predictions agree',
               'methods': rows}
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2)
    return {'dataset': name, 'fold': fold, 'saving': 1 - rows['shared_mi']['mean_acquired_fraction']}


def summarize():
    means = {}
    for name in DATASETS:
        folds = [json.loads((DEST / f'{name}-s9191-f{f}.json').read_text(encoding='utf-8'))
                 for f in range(3)]
        means[name] = {}
        for method in ['full', 'shared_mi', 'shared_adaptive', 'independent_mi', 'static_mi']:
            keys = ['BA', 'macro_F1', 'mean_acquired_fraction']
            if method in ['shared_mi', 'shared_adaptive', 'independent_mi']:
                keys += ['early_stop_fraction', 'full_over_conditional_speedup', 'cache_bytes', 'policy_setup_seconds']
            means[name][method] = {k: float(np.mean([f['methods'][method][k] for f in folds])) for k in keys}
    aggregate = {m: {k: float(np.mean([means[n][m][k] for n in DATASETS]))
                     for k in means[DATASETS[0]][m]} for m in means[DATASETS[0]]}
    saving = 1 - aggregate['shared_mi']['mean_acquired_fraction']
    improvement = (aggregate['independent_mi']['mean_acquired_fraction']
                   - aggregate['shared_mi']['mean_acquired_fraction'])
    gate = {'zero_disagreement_or_failure': True, 'mean_saving_at_least_20pct': saving >= .2,
            'at_least_four_datasets_save_10pct': sum(
                1 - means[n]['shared_mi']['mean_acquired_fraction'] >= .1 for n in DATASETS) >= 4,
            'shared_extra_saving_at_least_5pp': improvement >= .05}
    payload = {'completed_splits': 18, 'dataset_means': means, 'aggregate': aggregate,
               'gate': gate, 'gate_passed': all(gate.values()), 'development_only': True,
               'accuracy_gain_over_full_is_impossible_for_prediction_preserving_rule': True,
               'timing_scope': 'online scalar queries; training/caches separately reported; no real fees'}
    with (DEST / 'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2)
    print(json.dumps({'aggregate': aggregate, 'gate': gate}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    if (DEST / 'summary.json').exists():
        raise RuntimeError('completed study preserved; inspect existing summary')
    jobs = [(n, f) for n in DATASETS for f in range(3)]
    failures = []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run, j): j for j in jobs}
        for future in as_completed(futures):
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception as exc:
                failures.append({'job': futures[future], 'error': repr(exc)})
                print(json.dumps(failures[-1]), flush=True)
    if failures:
        with (DEST / 'failures.json').open('x', encoding='utf-8') as stream:
            json.dump(failures, stream, indent=2)
        raise RuntimeError('pilot failures preserved; do not summarize partial successes')
    summarize()


if __name__ == '__main__':
    main()
