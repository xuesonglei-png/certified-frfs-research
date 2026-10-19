"""Replay complete optimization and independently recompute trace constraints."""
import json
import hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose_vote_aligned_refinement import runner, fingerprints
from certified_frfs.datasets import load_dataset
from certified_frfs.vote_aligned_refinement import refine_vote_aligned
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.validated_pool import preprocess_fit, validation_scores


def numerical_equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            numerical_equal(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, z in zip(a, b):
            numerical_equal(x, z)
    elif isinstance(a, float):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)
    else:
        assert a == b


def main(objective_descent=True, support_exchange=False):
    threadpool_limits(1)
    counts = {'splits': 0, 'accepted_updates': 0, 'replayed_attempts': 0}
    for path in sorted(runner.FOLDER.glob('*.json')):
        if path.name.startswith('complete'):
            continue
        p = json.loads(path.read_text()); name = p['dataset']
        for relative, digest in p['fingerprints'].items():
            assert hashlib.sha256((runner.ROOT/relative).read_bytes()).hexdigest() == digest, relative
        prior = json.loads((runner.ROOT/'results/long-capped-sparse'/path.name).read_text())
        X, y = load_dataset(name)
        for index, s in enumerate(p['splits']):
            old = prior['splits'][index]
            ri, vi = np.array(s['reference_indices']), np.array(s['validation_indices'])
            assert ri.tolist() == old['reference_indices'] and vi.tolist() == old['validation_indices']
            assert set(ri).isdisjoint(vi)
            initial = old['fit']['compact_selected']
            fit = refine_vote_aligned(X[ri], y[ri], initial)
            assert fit == s['fit']
            A, B = preprocess_fit(X[ri], X[vi]); d = X.shape[1]; mass = np.sqrt(d)
            for node in fit['trace']:
                w = np.array(node['weights']); fs = np.flatnonzero(w > 0).tolist()
                if support_exchange:
                    assert len(fs) <= len(initial['features'])
                else:
                    assert set(fs) <= set(initial['features'])
                assert np.isfinite(w).all() and np.all(w >= 0)
                assert abs(w.sum()-mass) < 1e-8 and np.max(w) <= 2*mass/d+1e-8
                cert = weight_certificate(A, y[ri], fs, np.ones(d)/np.sqrt(d), w[fs])
                numerical_equal(cert, node['certificate'])
                assert cert['feasible']
            if objective_descent:
                assert np.all(np.diff([n['objective'] for n in fit['trace']]) < 0)
            assert np.all(np.diff([n['training_utility'] for n in fit['trace']]) >= -1e-12)
            metrics = validation_scores(A, y[ri], B, y[vi], fit['features'], np.array(fit['weights']), 144+index, joint=True)
            assert all(abs(metrics[k]-s['metrics'][k]) < 1e-12 for k in metrics)
            counts['splits'] += 1
            counts['accepted_updates'] += len(fit['trace'])-1
            counts['replayed_attempts'] += len(fit['attempts'])
        print(name, 'full optimizer and constraints replayed', flush=True)
    assert counts['splits'] == 44
    counts['scope'] = 'numerical_optimizer_replay_not_independent_proof_or_directed_rounding'
    (runner.FOLDER/'complete-audit.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == '__main__':
    main()
