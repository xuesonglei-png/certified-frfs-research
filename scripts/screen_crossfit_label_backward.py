"""All-reference screen; internal query selection is not a risk certificate."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from review_box_iterations import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.crossfit_label_backward import CrossfitLabelLoss, crossfit_label_path

FOLDER = ROOT/'results/crossfit-label-screen'


def run(name):
    threadpool_limits(1)
    score_source = ROOT/'results/nested-conditional-feasible'/(name+'.json')
    label_source = ROOT/'results/nested-conditional-label'/(name+'.json')
    score = json.loads(score_source.read_text()); label = json.loads(label_source.read_text())
    X, y = load_dataset(name); records = []
    for s, t in zip(score['splits'], label['splits']):
        assert s['reference_indices'] == t['reference_indices']
        ri = np.array(s['reference_indices']); raw, labels = X[ri], y[ri]
        crossfit = crossfit_label_path(raw, labels); loss = CrossfitLabelLoss(raw, labels)
        paths = {'crossfit': crossfit, 'teacher_score': s['stage']['conditional_path'],
                 'training_label': t['stage']['conditional_path']}
        common = set.intersection(*(set(len(r['features']) for r in p['trace']) for p in paths.values()))
        matches = []
        for size in sorted(common):
            matches.append({'size': size, **{method: loss(next(r['features'] for r in path['trace'] if len(r['features']) == size))
                                            for method, path in paths.items()}})
        records.append({'reference_indices': ri.tolist(), 'crossfit_path': crossfit, 'same_count_losses': matches,
                        'removed': {method: 1-len(path['trace'][-1]['features'])/X.shape[1] for method, path in paths.items()}})
        print(name, 'reference-only crossfit path complete', flush=True)
    files = [Path(__file__), score_source, label_source, ROOT/'src/certified_frfs/crossfit_label_backward.py',
             ROOT/'research/crossfit-label-screen-protocol.md', ROOT/'data/processed'/(name+'.npz')]
    (FOLDER/(name+'.json')).write_text(json.dumps({'dataset': name, 'd': X.shape[1], 'references': records,
        'fingerprints': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((ROOT/'results/nested-conditional-label/complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            future.result()
    rows = []
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); ratios = []
        for r in saved['references']:
            matches = [m for m in r['same_count_losses'] if m['size'] < saved['d']]
            if matches:
                m = min(matches, key=lambda r:r['size'])
                ratios.append((m['teacher_score']-m['crossfit'])/max(m['teacher_score'], 1e-12))
        rows.append({'dataset': name, 'removed': {method: float(np.mean([r['removed'][method] for r in saved['references']]))
                    for method in ('crossfit', 'teacher_score', 'training_label')},
                    'relative_crossfit_loss_improvement': float(np.mean(ratios)) if ratios else None})
    valid = [r['relative_crossfit_loss_improvement'] for r in rows if r['relative_crossfit_loss_improvement'] is not None]
    gain = float(np.mean(valid)) if valid else 0.
    compression = float(np.mean([r['removed']['crossfit']-r['removed']['teacher_score'] for r in rows]))
    report = {'rows': rows, 'relative_crossfit_loss_improvement': gain, 'compression_gain_pp': 100*compression,
              'datasets_with_common_nonfull_size': len(valid), 'gate_passed': gain >= .1 and compression >= -.02,
              'scope': 'all15_all44_reference_internal_crossfit_selected_surrogate_not_independent_prediction_or_risk_certificate'}
    (FOLDER/'complete.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print({k:v for k,v in report.items() if k != 'rows'}, flush=True)
