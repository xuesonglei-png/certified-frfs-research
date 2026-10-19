"""Independent replay without importing the new selector or its bound routine."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from certified_frfs.datasets import load_dataset

DEST = ROOT / 'results/conditional-completion-v1'


def manually_scaled(X, train, test):
    raw = X[train]
    medians = np.nanmedian(raw, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0)
    filled = np.where(np.isnan(raw), medians, raw)
    low, high = filled.min(axis=0), filled.max(axis=0)
    denominator = high - low
    denominator[denominator < 10 * np.finfo(float).eps] = 1
    scale = 1 / denominator
    offset = -low * scale
    A = np.clip(filled * scale + offset, 0, 1)
    B = np.clip(np.where(np.isnan(X[test]), medians, X[test]) * scale + offset, 0, 1)
    return A, B


def replay(R, labels, q, order, method):
    n, d = R.shape
    chosen, dist = [], np.zeros(n)
    checks = 0
    while len(chosen) < d:
        checks += 1
        target = labels[int(np.argmin(dist))]
        enemy = np.flatnonzero(labels != target)
        friends = np.flatnonzero(labels == target)
        friends = friends[np.argsort(dist[friends], kind='stable')[:3]]
        missing = np.array(sorted(set(range(d)) - set(chosen)), dtype=int)
        if method == 'independent_mi':
            upper = dist[friends] + np.maximum(R[friends][:, missing], 1 - R[friends][:, missing]).sum(axis=1) / d
            low = dist[enemy].min() - upper.min() if len(enemy) else np.inf
            pair = None
        else:
            # Match multiplication order at roundoff-sensitive bottleneck ties.
            # Independent stop-bound recomputation below uses direct coordinate sums.
            weight = np.full(len(missing), 1 / d)
            penalty = cdist(R[enemy][:, missing] * weight, R[friends][:, missing] * weight, 'cityblock')
            lower = dist[enemy, None] - dist[friends][None, :] - penalty
            best = lower.max(axis=1)
            e = int(best.argmin())
            f = int(lower[e].argmax())
            low = best[e]
            pair = (enemy[e], friends[f])
        if low > 2e-10:
            if method != 'independent_mi':
                exact_observed = np.abs(R[:, chosen] - q[chosen]).sum(axis=1) / d
                pair_lower = np.empty((len(enemy), len(friends)))
                for j, friend in enumerate(friends):
                    pair_lower[:, j] = (exact_observed[enemy] - exact_observed[friend]
                        - np.abs(R[enemy][:, missing] - R[friend, missing]).sum(axis=1) / d)
                assert pair_lower.max(axis=1).min() > 1e-10
            return chosen, int(target), True, checks
        if method == 'shared_adaptive' and chosen and pair is not None:
            scores = (1 / d) * np.abs(R[pair[0]] - R[pair[1]])
            scores[chosen] = -np.inf
            a = int(scores.argmax())
        else:
            a = int(next(a for a in order if a not in chosen))
        dist += (1 / d) * np.abs(R[:, a] - q[a])
        chosen.append(a)
    full = np.abs(R - q).mean(axis=1)
    return chosen, int(labels[full.argmin()]), False, checks


def main():
    threadpool_limits(1)
    start = time.perf_counter()
    audit_path = DEST / 'independent-audit.json'
    if audit_path.exists():
        raise RuntimeError('audit preserved; choose new audit filename')
    records = sorted(DEST.glob('*-s9191-f*.json'))
    assert len(records) == 18
    query_count = early = checked_metrics = 0
    result_hashes = {}
    for path in records:
        item = json.loads(path.read_text(encoding='utf-8'))
        name = item['dataset']
        X, y = load_dataset(name)
        splits = list(StratifiedKFold(3, shuffle=True, random_state=9191).split(X, y))
        tr, te = splits[item['fold']]
        assert item['train_indices'] == tr.tolist() and item['test_indices'] == te.tolist()
        assert not np.intersect1d(tr, te).size
        for relative, digest in item['source_hashes'].items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((ROOT / 'data/processed' / f'{name}.npz').read_bytes()).hexdigest() == item['data_sha256']
        A, B = manually_scaled(X, tr, te)
        teacher = y[tr][cdist(B, A, 'cityblock').argmin(axis=1)]
        assert teacher.tolist() == item['methods']['full']['predictions']
        for method in ['shared_mi', 'shared_adaptive', 'independent_mi']:
            row = item['methods'][method]
            predicted = []
            for query_id, (q, saved) in enumerate(zip(B, row['query_records'], strict=True)):
                chosen, pred, cert, checks = replay(A, y[tr], q, item['ranking'], method)
                assert chosen == saved['acquired'], (name, item['fold'], method, query_id, chosen, saved['acquired'])
                assert len(chosen) == saved['count'] == saved['cost']
                assert pred == saved['prediction']
                assert cert == saved['early_certified'] and checks == saved['bound_checks']
                predicted.append(pred)
                query_count += 1
                early += cert
            assert np.array_equal(predicted, teacher)
            for metric, actual in [('BA', balanced_accuracy_score(y[te], predicted)),
                                   ('macro_F1', f1_score(y[te], predicted, average='macro', zero_division=0))]:
                assert np.isclose(actual, row[metric], rtol=0, atol=1e-14)
                checked_metrics += 1
        fs = item['methods']['static_mi']['features']
        static = y[tr][cdist(B[:, fs], A[:, fs], 'cityblock').argmin(axis=1)]
        assert static.tolist() == item['methods']['static_mi']['predictions']
        result_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f'{name} fold {item["fold"]}: independent trajectory replay passed', flush=True)
    payload = {'passed': True, 'split_count': len(records), 'conditional_query_trajectories': query_count,
               'early_stop_certificates_replayed': int(early), 'metric_values_checked': checked_metrics,
               'result_sha256': result_hashes, 'audit_seconds': time.perf_counter() - start,
               'scope': 'manual preprocessing, independent scipy distances, all acquisition paths/stops/predictions',
               'limitations': 'floating-point replay is not interval arithmetic proof; no population accuracy claim',
               'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with audit_path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2)
    print(json.dumps({k: v for k, v in payload.items() if k != 'result_sha256'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
