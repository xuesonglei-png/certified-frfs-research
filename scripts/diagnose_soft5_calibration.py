"""All86 training-only soft/robust-hard/actual-hard neighborhood diagnostic."""
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from threadpoolctl import threadpool_limits
from run_guarded_soft5_descent_admission import ROOT, FOLDER as PARENT, digest
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from soft_topk_metric_objective import topk_leakage_bound

FOLDER = ROOT/'results/soft5-calibration-diagnostic'


def profile(A, y, actual_weights, *, normalization_mass=None):
    A, y, weights = np.asarray(A, float), np.asarray(y), np.asarray(actual_weights, float)
    if (A.ndim != 2 or len(A) < 7 or y.shape != (len(A),)
            or not np.isfinite(A).all() or A.min() < 0 or A.max() > 1
            or weights.shape != (A.shape[1],) or not np.isfinite(weights).all()
            or weights.min() < 0 or weights.sum() <= 0):
        raise ValueError('valid reference table with at least seven objects required')
    mass = weights.sum() if normalization_mass is None else normalization_mass
    if not np.isfinite(mass) or mass <= 0 or abs(weights.sum()-mass) > 1e-8:
        raise ValueError('matching original normalization mass required')
    normalized = weights/mass
    lo, hi = np.maximum(0, A-.01), np.minimum(1, A+.01)
    classes = np.unique(y); records, predictions = [], []
    for i in range(len(A)):
        ids = np.flatnonzero(np.arange(len(A)) != i); same = y[ids] == y[i]
        robust, clean = np.zeros(len(ids)), np.zeros(len(ids))
        for a, actual in enumerate(weights):
            near = np.maximum(0, np.maximum(lo[ids, a]-hi[i, a], lo[i, a]-hi[ids, a]))
            far = np.maximum(abs(lo[i, a]-hi[ids, a]), abs(hi[i, a]-lo[ids, a]))
            robust += normalized[a]*np.where(same, far, near)
            # EXACT stored-coordinate convention used by the actual classifier.
            clean += abs(A[ids, a]*actual-A[i, a]*actual)
        radius = brentq(lambda r: float(expit((r-robust)/.05).sum()-5),
                        float(robust.min()-2), float(robust.max()+2), xtol=1e-13, rtol=1e-14)
        q = expit((radius-robust)/.05); residual = float(abs(q.sum()-5)); assert residual <= 1e-10
        order = np.lexsort((ids, robust)); hard = np.zeros(len(ids), bool); hard[order[:5]] = True
        gap = float(robust[order[5]]-robust[order[4]])
        leakage = float(q[~hard].sum()/5); bound = topk_leakage_bound(gap, len(ids)) / 5
        own_soft, own_hard = float(q[same].sum()/5), float(same[order[:5]].mean())
        error = abs(own_soft-own_hard)
        assert error <= leakage+1e-9 and leakage <= bound+1e-9
        clean_order = np.lexsort((ids, clean))[:5]
        prediction = classes[np.argmax([np.sum(y[ids[clean_order]] == c) for c in classes])]
        predictions.append(prediction)
        records.append((leakage, bound, gap, own_soft, own_hard, error, residual))
    rows = np.array(records); correct = np.array(predictions) == y
    balanced = lambda values: float(np.mean([np.mean(values[y == c]) for c in classes]))
    return {'objects': len(A), 'clean_hard5_LOO_BA': balanced(correct),
            'balanced_robust_soft_own_mass': balanced(rows[:, 3]),
            'balanced_robust_hard5_own_fraction': balanced(rows[:, 4]),
            'mean_robust_soft_to_hard_leak_fraction': float(rows[:, 0].mean()),
            'mean_conditional_leak_bound_fraction': float(rows[:, 1].mean()),
            'median_robust_boundary_gap': float(np.median(rows[:, 2])),
            'fraction_positive_robust_boundary_gaps': float(np.mean(rows[:, 2] > 0)),
            'mean_robust_class_fraction_error': float(rows[:, 5].mean()),
            'max_robust_class_fraction_error': float(rows[:, 5].max()),
            'max_mass_residual': float(rows[:, 6].max()), 'temperature': .05,
            'scope': 'training_only_robust_surrogate_calibration_and_actual_clean_LOO_not_test_risk'}


def run(name):
    threadpool_limits(1); audit = verify(PARENT)
    saved = json.loads((PARENT/(name+'.json')).read_text())
    for rel, sha in saved['fingerprints'].items(): assert digest(ROOT/rel) == sha
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in
              (Path(__file__), ROOT/'scripts/soft_topk_metric_objective.py', PARENT/(name+'.json'),
               PARENT/'complete-audit.json', ROOT/'data/processed'/(name+'.npz'))}
    out = FOLDER/(name+'.json')
    if out.exists():
        previous = json.loads(out.read_text()); assert previous['fingerprints'] == hashes
        if previous['complete']: return previous
    X, y = load_dataset(name); ri = np.array(saved['reference_indices']); A, _ = preprocess_fit(X[ri], X[ri]); labels = y[ri]
    results = []
    for r in saved['records']:
        fs = r['features']; mass = r['fit']['mass']
        anchor = profile(A[:, fs], labels, r['anchor'], normalization_mass=mass)
        target = profile(A[:, fs], labels, r['target'], normalization_mass=mass)
        results.append({'features': fs, 'anchor': anchor, 'target': target,
                        'verified_training_objective_decrease': r['fit']['training_objective_initial']-r['fit']['training_objective_final'],
                        'clean_hard5_LOO_gain_pp': 100*(target['clean_hard5_LOO_BA']-anchor['clean_hard5_LOO_BA'])})
    result = {'dataset': name, 'records': results, 'fingerprints': hashes, 'complete': True,
              'query_labels_or_accuracy_used': False, 'source_audit_sha256': digest(PARENT/'complete-audit.json')}
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(name, 'training calibration complete', len(results), flush=True)
    return result


def main():
    FOLDER.mkdir(exist_ok=True); names = sorted(verify(PARENT)['artifact_hashes'])
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = [f.result() for f in as_completed([executor.submit(run, n) for n in names])]
    counts, losses, rows = Counter(), [], []
    for r in sorted(results, key=lambda x: x['dataset']):
        for record in r['records']:
            gain = record['clean_hard5_LOO_gain_pp']; counts['supports'] += 1
            counts['training_hard5_improved' if gain > 1e-8 else 'training_hard5_worsened' if gain < -1e-8 else 'training_hard5_tied'] += 1
            losses.append(gain)
        rows.append({'dataset': r['dataset'], 'supports': len(r['records']),
                     'mean_clean_hard5_LOO_gain_pp': float(np.mean([t['clean_hard5_LOO_gain_pp'] for t in r['records']]))})
    assert len(rows) == 15 and counts['supports'] == 86
    result = {'counts': dict(counts), 'rows': rows, 'mean_support_LOO_gain_pp': float(np.mean(losses)),
              'goal_achieved': False, 'query_labels_or_accuracy_used': False,
              'scope': 'training_diagnostic_not_test_advantage_or_optimizer_selection',
              'artifact_hashes': {r['dataset']: digest(FOLDER/(r['dataset']+'.json')) for r in results}}
    (FOLDER/'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
