"""Independent direct ridge costs and scalar certificate checks for all paths."""
import hashlib
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.triangular_classifier import triangular_memberships
from certified_frfs.feasible_transport import weight_certificate


def main(source_name='conditional-backward-screen'):
    threadpool_limits(1)
    folder = ROOT/'results'/source_name
    summary = json.loads((folder/'complete.json').read_text())
    counts = {'references': 0, 'adaptive_paths': 0, 'ranking_costs': 0, 'certificates': 0, 'common_counts': 0}
    for row in summary['rows']:
        name = row['dataset']
        saved = json.loads((folder/(name+'.json')).read_text())
        source = json.loads((folder.parent/'nested-balanced-joint'/(name+'.json')).read_text())
        X, y = load_dataset(name)
        for rel, digest in saved['fingerprints'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
        for record, prior in zip(saved['references'], source['splits']):
            assert record['reference_indices'] == prior['reference_indices']
            assert 'validation_indices' not in record
            counts['references'] += 1
            assert record['joint_feasible'] == prior['stage']['joint_fit']['feasible']
            if not record['joint_feasible']:
                continue
            ri = np.array(record['reference_indices'])
            A, _ = preprocess_fit(X[ri], X[ri]); labels = y[ri]; d = A.shape[1]
            W = np.array(prior['stage']['teacher_coef']); q = len(W)
            Z = triangular_memberships(A)
            target = np.einsum('ndk,qdk->nq', Z, W-W.mean(axis=0, keepdims=True))
            weights = np.array([1/(q*sum(labels == label)) for label in labels])
            cache = {}

            def loss(fs):
                key = tuple(fs)
                if key not in cache:
                    design = np.column_stack([np.ones(len(A)), Z[:, fs].reshape(len(A), -1)])
                    penalty = np.eye(design.shape[1])*.001; penalty[0, 0] = 0
                    beta = np.linalg.solve(design.T@(weights[:, None]*design)+penalty,
                                           design.T@(weights[:, None]*target))
                    cache[key] = float(weights@np.mean((design@beta-target)**2, axis=1))
                return cache[key]

            active = list(range(d)); path = record['adaptive']; attempt_index = 0; accepted_index = 1
            for ranking in path['rankings']:
                assert ranking['features'] == active
                choices = sorted((loss([a for a in active if a != removed]), removed) for removed in active)
                assert [r['removed'] for r in ranking['choices']] == [r[1] for r in choices]
                np.testing.assert_allclose([r['loss'] for r in ranking['choices']], [r[0] for r in choices], rtol=1e-8, atol=1e-10)
                counts['ranking_costs'] += len(choices)
                accepted = False
                for value, removed in choices[:path['checks']]:
                    attempt = path['attempts'][attempt_index]; attempt_index += 1
                    fs = [a for a in active if a != removed]
                    assert attempt['removed'] == removed and attempt['features'] == fs
                    cert = weight_certificate(A, labels, fs, np.ones(d)/np.sqrt(d), attempt['weights'])
                    assert cert == attempt['certificate'] and attempt['accepted'] == cert['feasible']
                    counts['certificates'] += 1
                    if cert['feasible']:
                        step = path['trace'][accepted_index]; accepted_index += 1
                        assert step['features'] == fs
                        np.testing.assert_allclose(np.array(step['weights'])[fs], attempt['weights'], atol=1e-12)
                        active = fs; accepted = True
                        break
                if not accepted:
                    assert ranking == path['rankings'][-1]
            assert attempt_index == len(path['attempts']) and accepted_index == len(path['trace'])
            assert active == path['trace'][-1]['features']
            for method in ('adaptive', 'static'):
                for step in record[method]['trace']:
                    fs = step['features']
                    np.testing.assert_allclose(step['reconstruction_loss'], loss(fs), rtol=1e-8, atol=1e-10)
                    assert weight_certificate(A, labels, fs, np.ones(d)/np.sqrt(d), np.array(step['weights'])[fs])['feasible']
                    counts['certificates'] += 1
            expected = sorted(set(len(r['features']) for r in path['trace']) &
                              set(len(r['features']) for r in record['static']['trace']))
            assert [r['size'] for r in record['same_count_losses']] == expected
            for comparison in record['same_count_losses']:
                for method in ('adaptive', 'static'):
                    step = next(r for r in record[method]['trace'] if len(r['features']) == comparison['size'])
                    np.testing.assert_allclose(comparison[method+'_loss'], loss(step['features']), rtol=1e-8, atol=1e-10)
                counts['common_counts'] += 1
            counts['adaptive_paths'] += 1
        print(name, 'direct conditional costs and original certificates audited', flush=True)
    assert counts['references'] == 44
    counts['scope'] = 'direct_ridge_costs_ranking_shortlist_acceptance_and_FRS_replay_no_prediction_claim'
    (folder/'complete-audit.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    print(counts, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='conditional-backward-screen')
    main(parser.parse_args().source)
