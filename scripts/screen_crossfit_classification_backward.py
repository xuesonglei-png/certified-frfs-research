"""Reference-internal classification objective, complete matched-size screen."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from review_box_iterations import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.crossfit_classification_backward import CrossfitClassificationLoss, crossfit_classification_path

FOLDER = ROOT/'results/crossfit-classification-screen'


def run(name):
    threadpool_limits(1)
    source = ROOT/'results/nested-conditional-feasible'/(name+'.json')
    mse_source = ROOT/'results/crossfit-label-screen'/(name+'.json')
    score = json.loads(source.read_text()); mse = json.loads(mse_source.read_text())
    X, y = load_dataset(name); records = []
    for s, t in zip(score['splits'], mse['references']):
        assert s['reference_indices'] == t['reference_indices']
        ri = np.array(s['reference_indices']); raw, labels = X[ri], y[ri]
        path = crossfit_classification_path(raw, labels); loss = CrossfitClassificationLoss(raw, labels)
        paths = {'classification': path, 'teacher_score': s['stage']['conditional_path'],
                 'crossfit_MSE': t['crossfit_path']}
        common = set.intersection(*(set(len(r['features']) for r in p['trace']) for p in paths.values()))
        matches = []
        for size in sorted(common):
            matches.append({'size': size, **{method: {'balanced_error': values[0], 'MSE': values[1]}
                for method, p in paths.items()
                for values in [loss(next(r['features'] for r in p['trace'] if len(r['features']) == size))]}})
        records.append({'reference_indices': ri.tolist(), 'classification_path': path, 'same_count_losses': matches,
                        'removed': {method: 1-len(p['trace'][-1]['features'])/X.shape[1] for method, p in paths.items()}})
        print(name, 'reference-only classification path complete', flush=True)
    files = [Path(__file__), source, mse_source, ROOT/'src/certified_frfs/crossfit_classification_backward.py',
             ROOT/'src/certified_frfs/crossfit_label_backward.py', ROOT/'research/crossfit-classification-screen-protocol.md',
             ROOT/'data/processed'/(name+'.npz')]
    (FOLDER/(name+'.json')).write_text(json.dumps({'dataset': name, 'd': X.shape[1], 'references': records,
        'fingerprints': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((ROOT/'results/crossfit-label-screen/complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            future.result()
    rows = []
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); gains = {'teacher_score': [], 'crossfit_MSE': []}
        for r in saved['references']:
            matches = [m for m in r['same_count_losses'] if m['size'] < saved['d']]
            if matches:
                m = min(matches, key=lambda r:r['size'])
                for method in gains:
                    gains[method].append(m[method]['balanced_error']-m['classification']['balanced_error'])
        rows.append({'dataset': name, 'removed': {method: float(np.mean([r['removed'][method] for r in saved['references']]))
                    for method in ('classification', 'teacher_score', 'crossfit_MSE')},
                    'balanced_error_improvement': {method: float(np.mean(values)) if values else None for method, values in gains.items()}})
    valid = [r for r in rows if r['balanced_error_improvement']['teacher_score'] is not None]
    gains = {method: float(np.mean([r['balanced_error_improvement'][method] for r in valid])) if valid else 0.
             for method in ('teacher_score', 'crossfit_MSE')}
    compression = float(np.mean([r['removed']['classification']-r['removed']['teacher_score'] for r in rows]))
    report = {'rows': rows, 'balanced_error_improvements': gains, 'compression_gain_pp': 100*compression,
              'datasets_with_common_nonfull_size': len(valid),
              'gate_passed': all(v >= .005 for v in gains.values()) and compression >= -.02 and len(valid) >= 10,
              'scope': 'all15_all44_reference_internal_selected_classification_surrogate_not_certified_logistic_prediction_advantage'}
    (FOLDER/'complete.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print({k:v for k,v in report.items() if k != 'rows'}, flush=True)
