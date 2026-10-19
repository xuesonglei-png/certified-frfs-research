"""Replay reference-only deletion ordering, rejection and certificates."""
import hashlib
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.joint_certified_backward import certified_backward_path


def main():
    threadpool_limits(1)
    folder = ROOT/'results/joint-certified-backward-screen'
    summary = json.loads((folder/'complete.json').read_text())
    counts = {'references': 0, 'paths': 0, 'attempts': 0, 'accepted': 0}
    for row in summary['rows']:
        name = row['dataset']
        saved = json.loads((folder/(name+'.json')).read_text())
        source = json.loads((folder.parent/'nested-balanced-joint'/(name+'.json')).read_text())
        X, y = load_dataset(name)
        for rel, digest in saved['fingerprints'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
        for record, prior in zip(saved['references'], source['splits']):
            ri = np.array(record['reference_indices'])
            assert record['reference_indices'] == prior['reference_indices']
            assert 'validation_indices' not in record
            joint = prior['stage']['joint_fit']
            assert record['joint_feasible'] == joint['feasible']
            if joint['feasible']:
                A, _ = preprocess_fit(X[ri], X[ri])
                rebuilt = certified_backward_path(A, y[ri], joint['ranking'])
                assert rebuilt == record['path']
                counts['paths'] += 1
                counts['attempts'] += len(rebuilt['attempts'])
                counts['accepted'] += len(rebuilt['trace'])-1
            counts['references'] += 1
        print(name, 'training path replay passed', flush=True)
    assert counts['references'] == 44
    counts['scope'] = 'complete_reference_only_path_and_certificate_replay_no_classifier_or_prediction_claim'
    (folder/'complete-audit.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    print(counts, flush=True)


if __name__ == '__main__':
    main()
