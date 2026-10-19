"""Rebuild unconstrained oracle objective, query loss and validation metrics."""
import hashlib
import json
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from audit_sparse_pilots import preprocessing
from certified_frfs.datasets import load_dataset
from certified_frfs.query_probability import QueryProbabilityObjective
from certified_frfs.validated_pool import validation_scores


def main():
    threadpool_limits(1); folder = ROOT/'results/feasible-query-ablation'; path = folder/'complete.json'; p = json.loads(path.read_text())
    for rel, expected in p['source_hashes'].items(): assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == expected
    counts = {'trace_objectives': 0, 'linear_oracles': 0, 'external_scores': 0, 'splits': 0}
    for r in p['rows']:
        name = r['dataset']; X, y = load_dataset(name); prior = ROOT/'results/feasible-query-long-diagnostic'/f'{name}.json'
        assert hashlib.sha256(prior.read_bytes()).hexdigest() == r['prior_sha256']
        assert hashlib.sha256((ROOT/'data/processed'/f'{name}.npz').read_bytes()).hexdigest() == r['data_sha256']
        saved = json.loads(prior.read_text())
        for index, s in enumerate(r['splits']):
            assert s['certified_metrics'] == saved['splits'][index]['metrics']
            ri, vi = np.array(s['reference_indices']), np.array(s['validation_indices']); assert not set(ri) & set(vi)
            assert ri.tolist() == saved['splits'][index]['reference_indices']; assert vi.tolist() == saved['splits'][index]['validation_indices']
            A, B = preprocessing(X[ri], X[vi]); fit = s['fit']; full = np.ones(r['d'])/np.sqrt(r['d']); mass = full.sum()
            objective = QueryProbabilityObjective(X[ri], y[ri]); assert objective.splits == fit['training_splits']
            assert fit['certificate'] is None and not fit['certified']; previous = np.inf
            for node in fit['trace']:
                v = np.array(node['weights']); assert np.all(v >= 0); assert abs(v.sum()-mass) < 1e-10
                value = objective(v/mass)[0]; assert abs(value-node['objective']) < 1e-10
                assert value < previous; previous = value; counts['trace_objectives'] += 1
            for oracle in fit['oracles']:
                v = np.array(oracle['weights']); costs = np.array(oracle['fit']['support_costs'])
                assert oracle['certificate'] is None
                assert abs(v.sum()-mass) < 1e-10
                if oracle['fit']['solver_success']:
                    value = costs@v/mass+.001*costs.min()*np.maximum(full-v, 0.).sum()/mass
                    # For this oracle's positive costs, every vertex candidate
                    # placing the full mass on one coordinate is feasible.
                    vertices = np.eye(r['d'])*mass
                    candidates = vertices@costs/mass+.001*costs.min()*np.maximum(full[None, :]-vertices, 0.).sum(axis=1)/mass
                    assert value <= candidates.min()+1e-8
                counts['linear_oracles'] += 1
            np.testing.assert_array_equal(fit['weights'], fit['trace'][-1]['weights'])
            assert fit['features'] == np.flatnonzero(np.array(fit['weights']) > 0).tolist()
            scores = validation_scores(A, y[ri], B, y[vi], fit['features'], np.array(fit['weights']), 144+index, joint=True)
            assert all(abs(scores[k]-s['metrics'][k]) < 1e-12 for k in scores)
            counts['external_scores'] += 2; counts['splits'] += 1
        print(name, 'ablation audited', flush=True)
    result = {**counts, 'result_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'scope': 'objective_oracle_vertex_comparison_score_reconstruction_not_optimizer_replay_or_test_generalization'}
    (folder/'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with zipfile.ZipFile(ROOT/'output/feasible-query-ablation-evidence.zip', 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for rel in p['source_hashes']: z.write(ROOT/rel, rel)
        for file in folder.glob('*.json'): z.write(file, str(file.relative_to(ROOT)))
        z.write(ROOT/'scripts/audit_feasible_query_ablation.py', 'scripts/audit_feasible_query_ablation.py')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
