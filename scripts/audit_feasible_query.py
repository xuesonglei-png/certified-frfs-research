"""Rebuild objectives, fixed certificates and held-out scores for every trace."""
import hashlib
import json
import zipfile
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from diagnose_feasible_query import FOLDER, NAMES, hashes
from audit_sparse_pilots import preprocessing
from certified_frfs.datasets import load_dataset
from certified_frfs.query_probability import QueryProbabilityObjective
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.joint_milp import pairwise_loss_envelope
from certified_frfs.validated_pool import validation_scores
from certified_frfs.core import gamma

ARCHIVE_NAME = 'feasible-query-generation-evidence.zip'

def main():
    threadpool_limits(1); counters = {'trace_metrics': 0, 'oracle_metrics': 0,
        'external_scores': 0, 'shared_table_samples': 0, 'splits': 0, 'observed_violations': 0}
    for name in NAMES:
        path = FOLDER/f'{name}.json'; p = json.loads(path.read_text()); assert p['source_hashes'] == hashes()
        X, y = load_dataset(name)
        assert p['data_sha256'] == hashlib.sha256((ROOT/'data/processed'/f'{name}.npz').read_bytes()).hexdigest()
        prior = ROOT/'results/guarded-support-diagnostic'/f'{name}.json'
        assert p['prior_sha256'] == hashlib.sha256(prior.read_bytes()).hexdigest()
        allowed = set(i for s in p['splits'][:1] for i in s['reference_indices']+s['validation_indices'])
        for index, s in enumerate(p['splits']):
            ri, vi = np.array(s['reference_indices']), np.array(s['validation_indices'])
            assert not set(ri) & set(vi); assert set(ri) | set(vi) == allowed
            A, B = preprocessing(X[ri], X[vi]); labels = y[ri]; full = np.ones(p['d'])/np.sqrt(p['d']); mass = full.sum()
            fit = s['fit']; objective = QueryProbabilityObjective(X[ri], labels)
            assert objective.splits == fit['training_splits']
            for split in fit['training_splits']: assert not set(split['reference']) & set(split['query'])
            workspace = MajorityTerms(A, labels, weights=full)
            info = fit['oracles'][0]['fit'] if fit['oracles'] else None
            C = []
            if info is not None:
                protected = workspace.bounds(range(p['d']), allowed_enemies=2) > 1e-10
                assert all(protected[i] for i in info['targets'])
                for q, required in zip(np.unique(labels), info['required']): assert sum(labels[i] == q for i in info['targets']) >= required
                for ex in info['exceptions']:
                    i = ex['anchor']; enemies = np.setdiff1d(np.flatnonzero(labels != labels[i]), ex['excluded_enemies'])
                    C.append(workspace.terms[:, i, enemies, :].transpose(1, 2, 0).reshape(-1, p['d']))
            C = np.concatenate(C) if C else np.empty((0, p['d']))
            before = np.inf
            nodes = [(c['weights'], c['objective'], False) for c in fit['trace']]
            nodes += [(o['weights'], None, True) for o in fit['oracles']]
            for weights, recorded_value, is_oracle in nodes:
                v = np.array(weights); assert np.all(v >= 0); assert abs(v.sum()-mass) < 1e-10
                if len(C): assert np.all(C@v > 1e-10)
                bound = pairwise_loss_envelope(A, labels, full, v)['upper_bound']; assert bound <= .1
                if recorded_value is not None:
                    value = objective(v/mass)[0]; assert abs(value-recorded_value) < 1e-10
                    assert value < before; before = value; counters['trace_metrics'] += 1
                else: counters['oracle_metrics'] += 1
                rng = np.random.default_rng(1811)
                for _ in range(3):
                    Z = np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1)
                    assert gamma(Z, labels, weights=full)-gamma(Z, labels, weights=v) <= bound+1e-10
                    if info is not None:
                        D = cdist(Z*v, Z*v, 'cityblock'); np.fill_diagonal(D, np.inf)
                        nn = np.argsort(D, axis=1, kind='stable')[:, :5]
                        assert np.all(((labels[nn] == labels[:, None]).sum(axis=1) >= 3)[info['targets']])
                    counters['shared_table_samples'] += 1
            final = np.array(fit['weights']); np.testing.assert_array_equal(final, fit['trace'][-1]['weights'])
            assert fit['features'] == np.flatnonzero(final > 0).tolist()
            scores = validation_scores(A, labels, B, y[vi], fit['features'], final, 144+index, joint=True)
            assert all(abs(scores[k]-s['metrics'][k]) < 1e-12 for k in scores)
            counters['external_scores'] += 2; counters['splits'] += 1
        print(name, 'feasible-query audit complete', flush=True)
    result = {**counters, 'scope': 'objective_split_certificate_score_reconstruction_sampling_not_independent_optimizer_replay_or_test_risk'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with zipfile.ZipFile(ROOT/'output'/ARCHIVE_NAME, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for rel in hashes(): z.write(ROOT/rel, rel)
        for p in FOLDER.glob('*.json'): z.write(p, str(p.relative_to(ROOT)))
        z.write(ROOT/'scripts/audit_feasible_query.py', 'scripts/audit_feasible_query.py')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
