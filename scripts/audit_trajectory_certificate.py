"""Rebuild proposals, candidate certificates, selection and held-out scores."""
import hashlib
import json
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from diagnose_trajectory_certificate import FOLDER, hashes
from audit_sparse_pilots import preprocessing
from certified_frfs.datasets import load_dataset
from certified_frfs.query_probability import QueryProbabilityObjective
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.core import gamma


def main():
    threadpool_limits(1); prior_path = ROOT/'results/capped-query-diagnostic/complete.json'; prior = json.loads(prior_path.read_text())
    counts = {'candidates': 0, 'sampled_loss_checks': 0, 'external_scores': 0, 'splits': 0, 'observed_violations': 0}
    for old in prior['rows']:
        path = FOLDER/f"{old['dataset']}.json"; p = json.loads(path.read_text()); assert p['source_hashes'] == hashes()
        X, y = load_dataset(old['dataset']); full = np.ones(p['d'])/np.sqrt(p['d']); mass = full.sum()
        assert p['data_sha256'] == hashlib.sha256((ROOT/'data/processed'/f"{old['dataset']}.npz").read_bytes()).hexdigest()
        control_path = ROOT/'results/certified-capped-query-diagnostic'/f"{old['dataset']}.json"
        assert p['polytope_prior_sha256'] == hashlib.sha256(control_path.read_bytes()).hexdigest()
        for index, s in enumerate(p['splits']):
            original = old['splits'][index]; assert s['reference_indices'] == original['reference_indices']; assert s['validation_indices'] == original['validation_indices']
            ri, vi = np.array(s['reference_indices']), np.array(s['validation_indices']); assert not set(ri) & set(vi)
            A, B = preprocessing(X[ri], X[vi]); labels = y[ri]; objective = QueryProbabilityObjective(X[ri], labels); fit = s['fit']
            assert objective.splits == fit['training_splits']
            proposals = [tuple(full)]
            for node in original['fit']['trace'][1:]:
                for alpha in (.25, .5, .75, 1.):
                    proposals.append(tuple((1-alpha)*full+alpha*np.array(node['weights'])))
            assert set(proposals) == {tuple(c['weights']) for c in fit['candidates']}
            for c in fit['candidates']:
                v = np.array(c['weights']); fs = c['features']; assert abs(v.sum()-mass) < 1e-10
                assert np.all(v >= 0) and np.all(v <= 2*mass/p['d']+1e-10)
                assert fs == np.flatnonzero(v > 0).tolist()
                assert abs(objective(v/mass)[0]-c['objective']) < 1e-10
                cert = weight_certificate(A, labels, fs, full, v[fs]); saved = c['certificate']
                assert cert['feasible'] == saved['feasible']; assert cert['retained_counts'] == saved['retained_counts']
                assert cert['required'] == saved['required']
                bound = cert['dependency_certificate']['upper_bound']; assert abs(bound-saved['dependency_certificate']['upper_bound']) < 1e-12
                rng = np.random.default_rng(2211)
                for _ in range(3):
                    Z = np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1)
                    assert gamma(Z, labels, weights=full)-gamma(Z, labels, fs, weights=v) <= bound+1e-10
                    counts['sampled_loss_checks'] += 1
                counts['candidates'] += 1
            key = lambda c: (c['objective'], len(c['features']), c['iteration'], c['alpha'])
            assert fit['selected'] == min([c for c in fit['candidates'] if c['certificate']['feasible']], key=key)
            assert fit['unconstrained_selected'] == min(fit['candidates'], key=key)
            from certified_frfs.validated_pool import validation_scores
            for branch, name in [('certified', 'selected'), ('unconstrained', 'unconstrained_selected')]:
                c = fit[name]; scores = validation_scores(A, labels, B, y[vi], c['features'], np.array(c['weights']), 144+index, joint=True)
                assert all(abs(scores[k]-s['metrics'][branch][k]) < 1e-12 for k in scores); counts['external_scores'] += 2
            counts['splits'] += 1
        print(p['dataset'], 'trajectory candidate audit complete', flush=True)
    result = {**counts, 'prior_sha256': hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        'scope': 'all_candidate_certificate_objective_proposal_score_rebuild_sampling_not_optimizer_refit_or_rounding_proof'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with zipfile.ZipFile(ROOT/'output/trajectory-certificate-evidence.zip', 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for rel in hashes(): z.write(ROOT/rel, rel)
        for f in FOLDER.glob('*.json'): z.write(f, str(f.relative_to(ROOT)))
        z.write(ROOT/'scripts/audit_trajectory_certificate.py', 'scripts/audit_trajectory_certificate.py')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
