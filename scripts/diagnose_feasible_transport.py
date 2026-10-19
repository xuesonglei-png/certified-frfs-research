"""Frozen five-subset training diagnostic; infeasibility is not hidden."""
import json, hashlib, time, argparse
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.redistributed_search import CrossfitRedistributedUtilityWorkspace
from certified_frfs.feasible_transport import fit_feasible_transport, fit_repaired_transport
from certified_frfs.redistributed_loss import transfer_weights


def score(A, B, yt, yv, fs, w, index):
    rng = np.random.default_rng(144+index)
    Z = np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1)
    Q = np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1)
    labels = np.unique(yt); values = []
    for ref, query in [(A, B), (Z, Q)]:
        D = abs(query[:, None, fs]-ref[None, :, fs])@w
        nn = np.argsort(D, axis=1, kind='stable')[:, :min(5, len(yt))]
        votes = np.stack([(yt[nn] == q).sum(axis=1) for q in labels], axis=1)
        pred = labels[votes.argmax(axis=1)]
        values.append(float(np.mean([(pred[yv == q] == q).mean() for q in np.unique(yv)])))
    return {'clean': values[0], 'noisy': values[1], 'utility': float(np.mean(values))}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--repair-loo', action='store_true'); parser.add_argument('--repair-crossfit', action='store_true'); args = parser.parse_args()
    if args.repair_loo and args.repair_crossfit: raise ValueError('choose one selector')
    threadpool_limits(1)
    directory = ROOT/'results'/('repaired-crossfit-transport-diagnostic' if args.repair_crossfit else 'repaired-loo-transport-diagnostic' if args.repair_loo else 'feasible-transport-diagnostic'); directory.mkdir(exist_ok=True)
    out = directory/'training-subfold-gate.json'
    if out.exists(): raise FileExistsError(out)
    source = ROOT/'results/redistributed-metric-diagnostic/fold0-training.json'
    records = json.loads(source.read_text())['rows']; rows = []
    for r in records:
        started = time.perf_counter()
        o = json.loads((ROOT/'results/predictive-iteration36'/f"{r['dataset']}-s101-f0.json").read_text())
        family = next(p for p in o['rows'] if p['method'] == 'protected')['chosen']['family']
        X, y = load_dataset(r['dataset']); tr = np.array(o['train_indices']); fs = r['features']
        workspace = CrossfitRedistributedUtilityWorkspace(X[tr], y[tr], family=family)
        folds = []
        for index, (A, B, yt, yv, w) in enumerate(workspace.folds):
            repair = None; retained = fs
            if args.repair_loo or args.repair_crossfit:
                retained, v, repair = fit_repaired_transport(A, yt, fs, w, selection='crossfit' if args.repair_crossfit else 'loo', family=family)
                fit = repair['trace'][-1]['fit']
                # Full fallback requires the explicit original metric;
                # do not mislabel a failed grid as a certified selection.
            else:
                v, fit = fit_feasible_transport(A, yt, fs, w)
            # Identical candidates and objective: unconstrained optimum from
            # the already fitted grid, never a different classifier/search.
            key = (lambda q: (-q['selection_utility'], q['interpolation'])) if args.repair_crossfit else (lambda q: (-q['training_utility'], q['training_objective'], q['interpolation'])) if args.repair_loo else (lambda q: (q['training_objective'], q['interpolation']))
            unconstrained = min(fit['candidates'], key=key)
            folds.append({'fit': fit, 'repair': repair, 'retained_features': retained,
                'correlation': score(A, B, yt, yv, fs, transfer_weights(A, fs, w), index),
                'matched_correlation': score(A, B, yt, yv, retained, transfer_weights(A, retained, w), index),
                'class_endpoint': score(A, B, yt, yv, retained, np.asarray(fit['candidates'][-1]['weights']), index),
                'unconstrained': score(A, B, yt, yv, retained, np.asarray(unconstrained['weights']), index),
                'certified': None if v is None else score(A, B, yt, yv, retained, v, index)})
            print(r['dataset'], 'subfold', index, fit['status'], flush=True)
        complete = all(f['certified'] is not None for f in folds)
        delta = float(np.mean([f['certified']['utility']-f['correlation']['utility'] for f in folds])) if complete else None
        rows.append({'dataset': r['dataset'], 'features': fs, 'family': family,
            'folds': folds, 'splits': workspace.splits, 'complete_feasibility': complete,
            'utility_delta': delta, 'seconds': time.perf_counter()-started})
        # Recoverable checkpoint, not a completed result.
        (directory/'partial.json').write_text(json.dumps({'complete': False, 'rows': rows}, indent=2), encoding='utf-8')
    all_feasible = all(r['complete_feasibility'] for r in rows)
    mean = float(np.mean([r['utility_delta'] for r in rows])) if all_feasible else None
    gates = {'all_fifteen_subfolds_feasible': all_feasible,
             'mean_gain_at_least_half_point': mean is not None and mean >= .005,
             'at_least_three_positive': sum(r['utility_delta'] is not None and r['utility_delta'] > 1e-12 for r in rows) >= 3}
    payload = {'complete': True, 'scope': 'reused_training_fixed_subsets_not_test_advantage',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'rows': rows,
        'gates': gates, 'gate_passed': all(gates.values()), 'mean_utility_delta': mean}
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8'); print(json.dumps(gates), flush=True)


if __name__ == '__main__': main()
