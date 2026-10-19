"""Fixed paired-fold variability penalty on a frozen support pool.

The standard-error penalty is a selection heuristic, not a confidence bound:
crossfit folds overlap in their reference observations and search is adaptive.
"""
import hashlib
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from diagnose_joint_support import FOLDER, NAMES
from certified_frfs.cached_transfer_utility import CachedCrossfitTransferUtility
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit, validation_scores
from certified_frfs.redistributed_loss import transfer_weights
from certified_frfs.feasible_transport import weight_certificate


def paired_score(values, baseline):
    values, baseline = np.asarray(values, float), np.asarray(baseline, float)
    if values.ndim != 1 or values.shape != baseline.shape or len(values) < 2:
        raise ValueError('matching repeated fold utilities required')
    if not np.isfinite(values).all() or not np.isfinite(baseline).all():
        raise ValueError('finite utilities required')
    gains = values-baseline
    penalty = float(np.std(gains, ddof=1)/np.sqrt(len(gains)))
    return {'mean_gain': float(gains.mean()), 'variability_penalty': penalty,
            'penalized_gain': float(gains.mean()-penalty)}


def fold_values(workspace, features):
    fs = tuple(features); values = []
    for transfer, components, yt, yv in workspace.frames:
        weights = transfer.weights(fs); labels = np.unique(yt); scores = []
        for c in components:
            distances = c[:, :, fs]@weights
            neighbors = np.argsort(distances, axis=1, kind='stable')[:, :min(5, len(yt))]
            votes = np.stack([(yt[neighbors] == q).sum(axis=1) for q in labels], axis=1)
            pred = labels[votes.argmax(axis=1)]
            scores.append(float(np.mean([(pred[yv == q] == q).mean() for q in np.unique(yv)])))
        values.append(float(np.mean(scores)))
    return np.array(values)


def main():
    threadpool_limits(1)
    target = ROOT/'results/paired-stability-diagnostic'
    target.mkdir(exist_ok=True)
    output = target/'complete.json'
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for name in NAMES:
        source = FOLDER/f'{name}.json'; p = json.loads(source.read_text())
        X, y = load_dataset(name); splits = []; full_fs = tuple(range(p['d']))
        for index, s in enumerate(p['splits']):
            ri, vi = np.array(s['reference_indices']), np.array(s['validation_indices'])
            A, B = preprocess_fit(X[ri], X[vi]); full = np.ones(p['d'])/np.sqrt(p['d'])
            workspace = CachedCrossfitTransferUtility(X[ri], y[ri])
            baseline = fold_values(workspace, full_fs)
            candidates = []
            for c in s['fit']['pool']:
                values = fold_values(workspace, c['features'])
                assert abs(values.mean()-c['utility']) < 1e-12
                candidates.append({**c, 'fold_utilities': values.tolist(), **paired_score(values, baseline)})
            candidates.sort(key=lambda c: (-c['penalized_gain'], -c['utility'],
                tuple(c['features']) != full_fs, len(c['features']), c['features']))
            shortlist = [c for c in candidates if tuple(c['features']) != full_fs][:12]
            shortlist.append(next(c for c in candidates if tuple(c['features']) == full_fs))
            certs = {}
            for c in shortlist:
                fs = c['features']
                certs[tuple(fs)] = weight_certificate(A, y[ri], fs, full, transfer_weights(A, fs, full))
            assert certs[full_fs]['feasible']
            feasible = [c for c in candidates if certs.get(tuple(c['features']), {}).get('feasible', False)]
            chosen = {'certified': feasible[0], 'unconstrained': candidates[0]}
            metrics = {}; fits = {}
            for branch, c in chosen.items():
                fs = c['features']; wide = np.zeros(p['d']); wide[fs] = transfer_weights(A, fs, full)
                metrics[branch] = validation_scores(A, y[ri], B, y[vi], fs, wide, 144+index, joint=True)
                fits[branch] = {**c, 'weights': wide.tolist(), 'certificate': certs.get(tuple(fs))}
            splits.append({'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(),
                'candidates': candidates, 'fits': fits, 'metrics': metrics,
                'certificates': [{'features': list(fs), **c} for fs, c in certs.items()],
                'reference_metrics': s['metrics']})
        rows.append({'dataset': name, 'd': p['d'], 'splits': splits,
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
        print(name, 'certified utility delta full', np.mean([
            (s['metrics']['certified']['clean']+s['metrics']['certified']['noise01']-
             s['reference_metrics']['full']['clean']-s['reference_metrics']['full']['noise01'])/2 for s in splits]), flush=True)
    def utility(s):
        return (s['clean']+s['noise01'])/2
    means = {m: float(np.mean([np.mean([utility(s['metrics'][m]) for s in r['splits']]) for r in rows]))
        for m in ('certified', 'unconstrained')}
    means.update({m: float(np.mean([np.mean([utility(s['reference_metrics'][m]) for s in r['splits']]) for r in rows]))
        for m in ('full', 'fixed_majority41')})
    positive = int(sum(np.mean([utility(s['metrics']['certified'])-utility(s['reference_metrics']['full']) for s in r['splits']]) > 1e-12 for r in rows))
    gates = {'gain_full_half_point': bool(means['certified']-means['full'] >= .005),
        'gain_fixed41_half_point': bool(means['certified']-means['fixed_majority41'] >= .005),
        'ten_dataset_improvements_full': bool(positive >= 10)}
    result = {'rows': rows, 'means': means, 'gates': gates, 'gate_passed': all(gates.values()),
        'positive_datasets_full': positive, 'penalty_coefficient': 1.,
        'script_sha256': hashlib.sha256((ROOT/'scripts/diagnose_paired_stability.py').read_bytes()).hexdigest(),
        'scope': 'heuristic_on_reused_pool_all_inner_development_not_confidence_bound_or_test_superiority'}
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
