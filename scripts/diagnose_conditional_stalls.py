"""Check excluded candidates at stopped paths; no prediction or retuning."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.redistributed_loss import transfer_weights
from certified_frfs.capped_sparse import capped_sparse_projection
from certified_frfs.feasible_transport import weight_certificate

SOURCE = ROOT/'results/conditional-backward-screen'
FOLDER = ROOT/'results/conditional-stall-diagnostic'


def run(name):
    threadpool_limits(1)
    source = SOURCE/(name+'.json'); saved = json.loads(source.read_text())
    X, y = load_dataset(name); records = []
    for record in saved['references']:
        if not record['joint_feasible']:
            continue
        path = record['adaptive']; final = path['trace'][-1]; fs = final['features']; d = saved['d']
        if len(fs) <= int(np.ceil(d/2)):
            continue
        ranking = path['rankings'][-1]
        assert ranking['features'] == fs
        ri = np.array(record['reference_indices']); A, _ = preprocess_fit(X[ri], X[ri])
        mass = np.sqrt(d); original = np.ones(d)/mass; excluded = []
        for choice in ranking['choices'][3:]:
            support = [a for a in fs if a != choice['removed']]
            weights = capped_sparse_projection(transfer_weights(A, support, original)/mass, len(support), 2/d)*mass
            cert = weight_certificate(A, y[ri], support, original, weights)
            excluded.append({**choice, 'features': support, 'weights': weights.tolist(), 'certificate': cert})
        records.append({'reference_indices': ri.tolist(), 'stopped_features': fs,
                        'excluded_candidates': excluded})
    files = [source, Path(__file__), ROOT/'data/processed'/(name+'.npz')]
    (FOLDER/(name+'.json')).write_text(json.dumps({'dataset': name, 'records': records,
        'fingerprints': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}, indent=2), encoding='utf-8')
    print(name, 'excluded stalled candidates checked', flush=True)


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            future.result()
    records = [r for name in names for r in json.loads((FOLDER/(name+'.json')).read_text())['records']]
    report = {'stopped_references': len(records), 'excluded_checks': sum(len(r['excluded_candidates']) for r in records),
              'stops_with_feasible_excluded_deletion': sum(any(c['certificate']['feasible'] for c in r['excluded_candidates']) for r in records),
              'feasible_excluded_candidates': sum(c['certificate']['feasible'] for r in records for c in r['excluded_candidates']),
              'scope': 'reference_only_stall_feasibility_diagnostic_not_selected_method_or_prediction'}
    (FOLDER/'complete.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(report, flush=True)
