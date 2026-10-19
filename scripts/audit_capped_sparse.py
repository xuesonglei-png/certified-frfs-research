"""Independent recomputation of sparse metric traces and certificates."""
import json
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose_capped_sparse import ROOT, FOLDER, fingerprints
from audit_trajectory_recent import scaling
from certified_frfs.datasets import load_dataset
from certified_frfs.query_probability import QueryProbabilityObjective
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.core import gamma
from certified_frfs.validated_pool import validation_scores
from certified_frfs.capped_sparse import capped_sparse_projection


def main():
    threadpool_limits(1); counts = {'candidates': 0, 'sparse_candidates': 0, 'trace_values': 0,
        'scores': 0, 'sampled_loss_checks': 0, 'splits': 0}
    for path in sorted(FOLDER.glob('*.json')):
        if path.name.startswith('complete'): continue
        p = json.loads(path.read_text()); name = p['dataset']; assert p['fingerprints'] == fingerprints(name)
        X, y = load_dataset(name); full = np.ones(p['d'])/np.sqrt(p['d']); mass = full.sum()
        prior = json.loads((ROOT/'results/trajectory-certificate-diagnostic'/f'{name}.json').read_text())
        for index, s in enumerate(p['splits']):
            ri, vi = np.array(s['reference_indices']), np.array(s['validation_indices']); old = prior['splits'][index]
            assert ri.tolist() == old['reference_indices'] and vi.tolist() == old['validation_indices']
            assert set(ri).isdisjoint(vi)
            A, B = scaling(X[ri], X[vi]); labels = y[ri]; obj = QueryProbabilityObjective(X[ri], labels)
            fit = s['fit']; assert fit['training_splits'] == obj.splits
            originals = [c for c in fit['candidates'] if c['origin']=='old_trajectory']
            assert [c['weights'] for c in originals] == [c['weights'] for c in old['fit']['candidates']]
            sparse = [c for c in fit['candidates'] if c['origin']=='capped_sparse']
            expected = sorted({max(int(np.ceil(p['d']/2)), int(np.ceil(f*p['d']))) for f in (.5, .65, .8)})
            assert [c['fit']['requested_cardinality'] for c in sparse] == expected
            initial = np.array(old['fit']['unconstrained_selected']['weights'])/mass
            for c in fit['candidates']:
                v = np.array(c['weights']); fs = c['features']
                assert abs(v.sum()-mass) < 1e-10 and v.min() >= 0 and v.max() <= 2*mass/p['d']+1e-10
                assert fs == np.flatnonzero(v > 0).tolist()
                assert abs(obj(v/mass)[0]-c['objective']) < 1e-10
                cert = weight_certificate(A, labels, fs, full, v[fs]); saved = c['certificate']
                for key in ('feasible', 'majority_feasible', 'retained_counts', 'required'): assert cert[key] == saved[key]
                bound = cert['dependency_certificate']['upper_bound']
                assert abs(bound-saved['dependency_certificate']['upper_bound']) < 1e-12
                Z = np.clip(A+np.random.default_rng(2727).uniform(-.01, .01, A.shape), 0, 1)
                assert gamma(Z, labels, weights=full)-gamma(Z, labels, fs, weights=v) <= bound+1e-10
                counts['sampled_loss_checks'] += 1; counts['candidates'] += 1
                if c['origin']=='capped_sparse':
                    k = c['fit']['requested_cardinality']; trace = c['fit']['trace']; last = np.inf
                    assert np.allclose(trace[0]['weights'], capped_sparse_projection(initial, k, 2/p['d']), atol=1e-12, rtol=0)
                    for node in trace:
                        w = np.array(node['weights']); value = obj(w)[0]
                        assert abs(value-node['objective']) < 1e-10 and value <= last+1e-12
                        assert np.count_nonzero(w) <= k and w.min() >= 0 and w.max() <= 2/p['d']+1e-12 and abs(w.sum()-1) < 1e-12
                        last = value; counts['trace_values'] += 1
                    assert np.allclose(v/mass, trace[-1]['weights'], atol=1e-12, rtol=0)
                    counts['sparse_candidates'] += 1
            eligible = [c for c in fit['candidates'] if c['certificate']['feasible']]
            exact = min(eligible, key=lambda c:(c['objective'], len(c['features'])))
            compact = min((c for c in eligible if c['objective'] <= exact['objective']+.01), key=lambda c:(len(c['features']), c['objective']))
            assert fit['tolerance'] == .01 and fit['selected'] == exact and fit['compact_selected'] == compact
            assert fit['unconstrained_selected'] == min(fit['candidates'], key=lambda c:(c['objective'], len(c['features'])))
            for branch,key in [('exact', 'selected'), ('compact', 'compact_selected'), ('unconstrained', 'unconstrained_selected')]:
                c = fit[key]; metrics = validation_scores(A, labels, B, y[vi], c['features'], np.array(c['weights']), 144+index, joint=True)
                assert all(abs(metrics[k]-s['metrics'][branch][k]) < 1e-12 for k in metrics)
                counts['scores'] += 2
            counts['splits'] += 1
        print(name, 'sparse extension audit complete', flush=True)
    result = {**counts, 'scope': 'independent_objective_certificate_projection_initialization_selection_score_rebuild_not_optimizer_refit_or_rounding_proof'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with zipfile.ZipFile(ROOT/'output/capped-sparse-extension-evidence.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in FOLDER.glob('*.json'): archive.write(path, str(path.relative_to(ROOT)))
        for path in (ROOT/'src/certified_frfs').glob('*.py'): archive.write(path, str(path.relative_to(ROOT)))
        for rel in ('scripts/diagnose_capped_sparse.py', 'scripts/audit_capped_sparse.py', 'tests/test_capped_sparse.py', 'research/capped-sparse-extension-protocol.md'):
            archive.write(ROOT/rel, rel)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
