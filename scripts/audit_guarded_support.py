"""Rebuild discovery utilities, guard scores, certificates and predictions."""
import hashlib
import json
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from diagnose_guarded_support import FOLDER, NAMES, hashes
from audit_sparse_pilots import preprocessing
from certified_frfs.datasets import load_dataset
from certified_frfs.cached_transfer_utility import CachedCrossfitTransferUtility
from certified_frfs.redistributed_loss import transfer_weights
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.core import gamma
from certified_frfs.validated_pool import validation_scores


def correctness(A, yt, B, yv, fs, w):
    labels = np.unique(yt); rng = np.random.default_rng(180); tables = [(A, B)]
    for _ in range(3):
        tables.append((np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1),
                       np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1)))
    outcomes = []
    for ref, query in tables:
        distance = np.abs(query[:, None, fs]-ref[None, :, fs])@w
        nn = np.argsort(distance, axis=1, kind='stable')[:, :min(5, len(yt))]
        counts = np.column_stack([(yt[nn] == q).sum(axis=1) for q in labels])
        outcomes.append(labels[np.argmax(counts, axis=1)] == yv)
    return np.mean(outcomes, axis=0)


def main():
    threadpool_limits(1)
    counts = {'discovery_scores': 0, 'guard_scores': 0, 'certificates': 0,
        'external_scores': 0, 'sampled_losses': 0, 'splits': 0, 'observed_violations': 0}
    records = []
    for name in NAMES:
        path = FOLDER/f'{name}.json'; p = json.loads(path.read_text())
        assert p['source_hashes'] == hashes()
        assert p['data_sha256'] == hashlib.sha256((ROOT/'data/processed'/f'{name}.npz').read_bytes()).hexdigest()
        assert p['prior_sha256'] == hashlib.sha256((ROOT/'results/joint-support-diagnostic'/f'{name}.json').read_bytes()).hexdigest()
        X, y = load_dataset(name); full = tuple(range(p['d'])); w = np.ones(p['d'])/np.sqrt(p['d'])
        for index, s in enumerate(p['splits']):
            fit = s['fit']; ri, vi = np.array(s['reference_indices']), np.array(s['validation_indices'])
            assert not set(ri) & set(vi)
            di, gi = np.array(fit['discovery_indices']), np.array(fit['guard_indices'])
            assert not set(di) & set(gi)
            assert set(di) | set(gi) == set(range(len(ri)))
            workspace = CachedCrossfitTransferUtility(X[ri][di], y[ri][di])
            assert workspace.splits == fit['discovery_crossfit_splits']
            for c in fit['pool']:
                actual = workspace.evaluate(c['features'])
                assert abs(actual['utility']-c['utility']) < 1e-12
                counts['discovery_scores'] += 1
            expected = [c['features'] for c in fit['pool'] if tuple(c['features']) != full][:12]+[list(full)]
            assert {tuple(c['features']) for c in fit['guarded_candidates']} == set(map(tuple, expected))
            A, B = preprocessing(X[ri], X[vi]); D, G = preprocessing(X[ri][di], X[ri][gi])
            baseline = correctness(D, y[ri][di], G, y[ri][gi], full, w)
            np.testing.assert_array_equal(baseline, fit['guard_baseline_correctness'])
            for c in fit['guarded_candidates']:
                fs = tuple(c['features']); wr = transfer_weights(D, fs, w)
                correct = correctness(D, y[ri][di], G, y[ri][gi], fs, wr)
                np.testing.assert_array_equal(correct, c['guard_correctness'])
                gain = correct-baseline; labs = y[ri][gi]; qs = np.unique(labs)
                mean = float(np.mean([gain[labs == q].mean() for q in qs]))
                variance = sum(float(np.var(gain[labs == q], ddof=1)/sum(labs == q)) if sum(labs == q) > 1 else 1. for q in qs)
                penalty = np.sqrt(variance)/len(qs)
                if fs == full:
                    mean = penalty = 0.
                assert abs(c['mean_gain']-mean) < 1e-12
                assert abs(c['penalized_gain']-mean+penalty) < 1e-12
                counts['guard_scores'] += 1
            certs = {}
            for c in fit['certificates']:
                fs = c['features']; wr = transfer_weights(A, fs, w)
                cert = weight_certificate(A, y[ri], fs, w, wr)
                assert cert['feasible'] == c['feasible']
                assert cert['retained_counts'] == c['retained_counts']
                bound = cert['dependency_certificate']['upper_bound']
                assert abs(bound-c['dependency_certificate']['upper_bound']) < 1e-12
                wide = np.zeros(p['d']); wide[fs] = wr
                rng = np.random.default_rng(1311)
                for _ in range(3):
                    Z = np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1)
                    loss = gamma(Z, y[ri], weights=w)-gamma(Z, y[ri], fs, weights=wide)
                    assert loss <= bound+1e-10
                    counts['sampled_losses'] += 1
                certs[tuple(fs)] = cert; counts['certificates'] += 1
            ranked = sorted(fit['guarded_candidates'], key=lambda c: (-c['penalized_gain'], -c['mean_gain'],
                tuple(c['features']) != full, len(c['features']), c['features']))
            assert ranked == fit['guarded_candidates']
            assert fit['unconstrained']['features'] == ranked[0]['features']
            feasible = [c for c in ranked if certs[tuple(c['features'])]['feasible']]
            assert fit['certified']['features'] == feasible[0]['features']
            for branch in ('certified', 'unconstrained'):
                c = fit[branch]; fs = c['features']; wide = np.array(c['weights'])
                np.testing.assert_allclose(wide[fs], transfer_weights(A, fs, w), rtol=0, atol=1e-12)
                assert abs(wide.sum()-w.sum()) < 1e-12
                metrics = validation_scores(A, y[ri], B, y[vi], fs, wide, 144+index, joint=True)
                assert all(abs(metrics[k]-s['metrics'][branch][k]) < 1e-12 for k in metrics)
                counts['external_scores'] += 2
            counts['splits'] += 1
        records.append({'dataset': name, 'result_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        print(name, 'guarded audit complete', flush=True)
    output = {**counts, 'records': records,
        'scope': 'independent_guard_prediction_arithmetic_certificate_rebuild_discovery_score_recompute_not_search_replay_or_box_proof'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    archive = ROOT/'output/guarded-support-generation-evidence.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for relative in hashes():
            z.write(ROOT/relative, relative)
        z.write(ROOT/'scripts/audit_guarded_support.py', 'scripts/audit_guarded_support.py')
        for file in sorted(FOLDER.glob('*.json')):
            z.write(file, str(file.relative_to(ROOT)))
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == '__main__':
    main()
