"""All44 reference-only feasibility screen; no outer prediction/labels used."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.joint_certified_backward import certified_backward_path

SOURCE = ROOT/'results/nested-balanced-joint'
FOLDER = ROOT/'results/joint-certified-backward-screen'


def run(name):
    threadpool_limits(1)
    source = SOURCE/(name+'.json')
    saved = json.loads(source.read_text())
    X, y = load_dataset(name)
    rows = []
    for split in saved['splits']:
        ri = np.array(split['reference_indices'])
        A, _ = preprocess_fit(X[ri], X[ri])
        joint = split['stage']['joint_fit']
        record = {'reference_indices': ri.tolist(), 'joint_feasible': joint['feasible']}
        if joint['feasible']:
            record['path'] = certified_backward_path(A, y[ri], joint['ranking'])
        rows.append(record)
    files = [source, Path(__file__), ROOT/'src/certified_frfs/joint_certified_backward.py',
             ROOT/'research/joint-certified-backward-screen-protocol.md', ROOT/'data/processed'/(name+'.npz')]
    output = {'dataset': name, 'd': X.shape[1], 'references': rows,
              'fingerprints': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    (FOLDER/(name+'.json')).write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(name, 'reference-only certified deletion screened', flush=True)


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            future.result()
    rows = []
    for name in names:
        result = json.loads((FOLDER/(name+'.json')).read_text())
        old = json.loads((SOURCE/(name+'.json')).read_text())
        removal, prior, admitted = [], [], []
        for reference, split in zip(result['references'], old['splits']):
            final = reference['path']['trace'][-1]['features'] if reference['joint_feasible'] else list(range(result['d']))
            removal.append(1-len(final)/result['d'])
            same = split['stage']['policies']['0.5']['candidate']['features']
            prior.append(1-len(same)/result['d'])
            admitted.append(len(reference['path']['trace'])-1 if reference['joint_feasible'] else 0)
        rows.append({'dataset': name, 'removed': float(np.mean(removal)),
                     'prior_fixed_half_removed': float(np.mean(prior)), 'accepted_deletions': sum(admitted)})
    improvement = float(np.mean([r['removed']-r['prior_fixed_half_removed'] for r in rows]))
    report = {'rows': rows, 'mean_removed': float(np.mean([r['removed'] for r in rows])),
              'mean_compression_gain_pp': 100*improvement, 'compression_gate_passed': improvement >= .05,
              'scope': 'all15_all44_reference_only_screen_no_prediction_or_outer_labels',
              'limitations': 'Compare fixed-half repaired metric supports only; no classification margin feasibility or predictive improvement established.'}
    (FOLDER/'complete.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print({k: v for k, v in report.items() if k != 'rows'}, flush=True)
