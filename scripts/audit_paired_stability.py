"""Audit saved reranking, endpoint selection and external predictions."""
import hashlib
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from audit_sparse_pilots import preprocessing
from certified_frfs.datasets import load_dataset
from certified_frfs.redistributed_loss import transfer_weights
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.validated_pool import validation_scores


def main():
    threadpool_limits(1)
    folder = ROOT/'results/paired-stability-diagnostic'
    source = folder/'complete.json'
    result = json.loads(source.read_text())
    assert result['script_sha256'] == hashlib.sha256((ROOT/'scripts/diagnose_paired_stability.py').read_bytes()).hexdigest()
    counters = {'candidate_ranks': 0, 'endpoint_certificates': 0, 'external_scores': 0, 'splits': 0}
    for r in result['rows']:
        name = r['dataset']; prior_path = ROOT/'results/joint-support-diagnostic'/f'{name}.json'
        assert r['source_sha256'] == hashlib.sha256(prior_path.read_bytes()).hexdigest()
        prior = json.loads(prior_path.read_text()); X, y = load_dataset(name)
        for index, s in enumerate(r['splits']):
            old = prior['splits'][index]
            assert s['reference_indices'] == old['reference_indices']
            assert s['validation_indices'] == old['validation_indices']
            assert {tuple(c['features']) for c in s['candidates']} == {tuple(c['features']) for c in old['fit']['pool']}
            full = next(c for c in s['candidates'] if len(c['features']) == r['d'])
            for c in s['candidates']:
                gains = np.array(c['fold_utilities'])-np.array(full['fold_utilities'])
                assert abs(c['mean_gain']-np.mean(gains)) < 1e-12
                assert abs(c['variability_penalty']-np.std(gains, ddof=1)/np.sqrt(len(gains))) < 1e-12
                assert abs(c['penalized_gain']-c['mean_gain']+c['variability_penalty']) < 1e-12
                counters['candidate_ranks'] += 1
            ordering = sorted(s['candidates'], key=lambda c: (-c['penalized_gain'], -c['utility'],
                len(c['features']) != r['d'], len(c['features']), c['features']))
            assert ordering == s['candidates']
            ri, vi = np.array(s['reference_indices']), np.array(s['validation_indices'])
            A, B = preprocessing(X[ri], X[vi]); w = np.ones(r['d'])/np.sqrt(r['d'])
            certificates = {}
            for c in s['certificates']:
                fs = c['features']; cert = weight_certificate(A, y[ri], fs, w, transfer_weights(A, fs, w))
                assert cert['feasible'] == c['feasible']
                assert abs(cert['dependency_certificate']['upper_bound']-c['dependency_certificate']['upper_bound']) < 1e-12
                certificates[tuple(fs)] = cert
                counters['endpoint_certificates'] += 1
            feasible = [c for c in ordering if certificates.get(tuple(c['features']), {}).get('feasible', False)]
            assert s['fits']['certified']['features'] == feasible[0]['features']
            assert s['fits']['unconstrained']['features'] == ordering[0]['features']
            for branch in ('certified', 'unconstrained'):
                c = s['fits'][branch]; fs = c['features']; weights = np.array(c['weights'])
                np.testing.assert_allclose(weights[fs], transfer_weights(A, fs, w), rtol=0, atol=1e-12)
                scores = validation_scores(A, y[ri], B, y[vi], fs, weights, 144+index, joint=True)
                assert all(abs(scores[k]-s['metrics'][branch][k]) < 1e-12 for k in scores)
                counters['external_scores'] += 2
            counters['splits'] += 1
        print(name, 'stability selection audited', flush=True)
    output = {**counters, 'result_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'scope': 'saved_fold_arithmetic_selection_certificate_prediction_rebuild_not_independent_fold_utility_recomputation'}
    (folder/'audit.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2), flush=True)


if __name__ == '__main__':
    main()
