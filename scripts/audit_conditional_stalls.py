"""Replay all excluded candidates and unchanged original metric certificates."""
import hashlib
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.redistributed_loss import transfer_weights
from certified_frfs.capped_sparse import capped_sparse_projection


def main():
    threadpool_limits(1)
    folder = ROOT/'results/conditional-stall-diagnostic'; summary = json.loads((folder/'complete.json').read_text())
    source_folder = folder.parent/'conditional-backward-screen'
    counts = {'stops': 0, 'checks': 0, 'feasible': 0, 'recoverable_stops': 0}
    for path in sorted(folder.glob('*.json')):
        if path.name.startswith('complete'):
            continue
        saved = json.loads(path.read_text()); name = saved['dataset']; X, y = load_dataset(name)
        source = json.loads((source_folder/path.name).read_text())
        byref = {tuple(r['reference_indices']): r for r in source['references']}
        for rel, digest in saved['fingerprints'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
        for record in saved['records']:
            original = byref[tuple(record['reference_indices'])]['adaptive']
            assert original['trace'][-1]['features'] == record['stopped_features']
            assert len(record['stopped_features']) > int(np.ceil(source['d']/2))
            choices = original['rankings'][-1]['choices'][3:]
            assert len(choices) == len(record['excluded_candidates'])
            ri = np.array(record['reference_indices']); A, _ = preprocess_fit(X[ri], X[ri]); d = A.shape[1]; mass = np.sqrt(d)
            recovered = False
            for expected, actual in zip(choices, record['excluded_candidates']):
                assert actual['removed'] == expected['removed'] and actual['loss'] == expected['loss']
                fs = [a for a in record['stopped_features'] if a != actual['removed']]
                assert actual['features'] == fs
                weights = capped_sparse_projection(transfer_weights(A, fs, np.ones(d)/mass)/mass, len(fs), 2/d)*mass
                np.testing.assert_allclose(weights, actual['weights'], atol=1e-12)
                cert = weight_certificate(A, y[ri], fs, np.ones(d)/mass, weights)
                assert cert == actual['certificate']
                counts['checks'] += 1; counts['feasible'] += cert['feasible']; recovered |= cert['feasible']
            counts['recoverable_stops'] += recovered; counts['stops'] += 1
        print(name, 'all excluded certificates replayed', flush=True)
    assert counts['stops'] == summary['stopped_references']
    assert counts['checks'] == summary['excluded_checks']
    assert counts['feasible'] == summary['feasible_excluded_candidates']
    assert counts['recoverable_stops'] == summary['stops_with_feasible_excluded_deletion']
    (folder/'complete-audit.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    print(counts, flush=True)


if __name__ == '__main__':
    main()
