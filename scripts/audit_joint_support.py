"""Rebuild scores/certificates for every endpoint and freeze source evidence."""
import hashlib
import json
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from diagnose_joint_support import FOLDER, NAMES, hashes
from audit_sparse_pilots import preprocessing
from certified_frfs.datasets import load_dataset
from certified_frfs.cached_transfer_utility import CachedCrossfitTransferUtility
from certified_frfs.redistributed_loss import transfer_weights
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.core import gamma
from certified_frfs.validated_pool import validation_scores


def main():
    threadpool_limits(1)
    totals = {'pool_scores': 0, 'endpoint_certificates': 0, 'external_scores': 0,
              'sampled_losses': 0, 'observed_violations': 0, 'splits': 0}
    for name in NAMES:
        path = FOLDER/f'{name}.json'
        p = json.loads(path.read_text())
        assert p['source_hashes'] == hashes()
        data = ROOT/'data/processed'/f'{name}.npz'
        assert hashlib.sha256(data.read_bytes()).hexdigest() == p['data_sha256']
        X, y = load_dataset(name)
        prior_path = ROOT/'results/predictive-iteration41'/f'{name}-s101-f0.json'
        assert hashlib.sha256(prior_path.read_bytes()).hexdigest() == p['prior_sha256']
        prior = json.loads(prior_path.read_text())
        allowed = set(prior['train_indices'])
        for index, s in enumerate(p['splits']):
            ri, vi = np.array(s['reference_indices']), np.array(s['validation_indices'])
            assert not set(ri) & set(vi)
            assert set(ri) | set(vi) == allowed
            A, B = preprocessing(X[ri], X[vi])
            fit = s['fit']
            assert fit['evaluations'] == len(fit['pool']) <= 64
            assert len({tuple(r['features']) for r in fit['pool']}) == len(fit['pool'])
            workspace = CachedCrossfitTransferUtility(X[ri], y[ri])
            assert workspace.splits == fit['search_splits']
            for c in fit['pool']:
                rebuilt = workspace.evaluate(c['features'])
                assert abs(rebuilt['utility']-c['utility']) < 1e-12
                totals['pool_scores'] += 1
            full = np.ones(p['d'])/np.sqrt(p['d'])
            certs = {}
            for c in fit['certificates']:
                fs = c['features']
                wr = transfer_weights(A, fs, full)
                cert = weight_certificate(A, y[ri], fs, full, wr)
                assert cert['feasible'] == c['feasible']
                assert cert['retained_counts'] == c['retained_counts']
                bound = cert['dependency_certificate']['upper_bound']
                assert abs(bound-c['dependency_certificate']['upper_bound']) < 1e-12
                certs[tuple(fs)] = cert
                wide = np.zeros(p['d']); wide[fs] = wr
                rng = np.random.default_rng(1103)
                for _ in range(3):
                    Z = np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1)
                    loss = gamma(Z, y[ri], weights=full)-gamma(Z, y[ri], fs, weights=wide)
                    assert loss <= bound+1e-10
                    totals['sampled_losses'] += 1
                totals['endpoint_certificates'] += 1
            eligible = [c for c in fit['pool'] if certs.get(tuple(c['features']), {}).get('feasible', False)]
            assert fit['certified']['features'] == eligible[0]['features']
            assert fit['unconstrained']['features'] == fit['pool'][0]['features']
            for branch in ('certified', 'unconstrained'):
                c = fit[branch]; fs = c['features']; wide = np.array(c['weights'])
                np.testing.assert_allclose(wide[fs], transfer_weights(A, fs, full), atol=1e-12, rtol=0)
                assert abs(wide.sum()-full.sum()) < 1e-12
                scores = validation_scores(A, y[ri], B, y[vi], fs, wide, 144+index, joint=True)
                assert all(abs(scores[k]-s['metrics'][branch][k]) < 1e-12 for k in scores)
                totals['external_scores'] += 2
            totals['splits'] += 1
        print(name, 'audited', flush=True)
    result = {**totals, 'scope': 'score_reconstruction_endpoint_refitting_shared_sampling_not_independent_search_or_rounding_proof'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    target = ROOT/'output/joint-support-generation-evidence.zip'
    if target.exists():
        raise FileExistsError(target)
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for path in hashes():
            z.write(ROOT/path, path)
        for path in sorted(FOLDER.glob('*.json')):
            z.write(path, str(path.relative_to(ROOT)))
        z.write(ROOT/'scripts/audit_joint_support.py', 'scripts/audit_joint_support.py')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
