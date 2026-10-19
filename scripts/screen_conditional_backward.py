"""All44 reference-only conditional-vs-static mechanism screen."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.conditional_certified_backward import conditional_backward_path, ReconstructionLoss
from certified_frfs.joint_certified_backward import certified_backward_path

SOURCE = ROOT/'results/nested-balanced-joint'
FOLDER = ROOT/'results/conditional-backward-screen'


def run(name):
    threadpool_limits(1)
    source = SOURCE/(name+'.json')
    saved = json.loads(source.read_text())
    X, y = load_dataset(name)
    records = []
    for index, split in enumerate(saved['splits']):
        ri = np.array(split['reference_indices'])
        A, _ = preprocess_fit(X[ri], X[ri])
        stage = split['stage']
        joint = stage['joint_fit']
        record = {'reference_indices': ri.tolist(), 'joint_feasible': joint['feasible']}
        if joint['feasible']:
            # Same original teacher scores for both adaptive and fixed paths.
            loss = ReconstructionLoss(A, y[ri], stage['teacher_coef'])
            adaptive = conditional_backward_path(A, y[ri], stage['teacher_coef'])
            static = certified_backward_path(A, y[ri], joint['ranking'])
            for r in static['trace']:
                r['reconstruction_loss'] = loss(r['features'])
            record.update({'adaptive': adaptive, 'static': static})
            common = sorted(set(len(r['features']) for r in adaptive['trace']) & set(len(r['features']) for r in static['trace']))
            comparisons = []
            for size in common:
                left = next(r for r in adaptive['trace'] if len(r['features']) == size)
                right = next(r for r in static['trace'] if len(r['features']) == size)
                comparisons.append({'size': size, 'adaptive_loss': left['reconstruction_loss'],
                                    'static_loss': right['reconstruction_loss']})
            record['same_count_losses'] = comparisons
        records.append(record)
        print(name, index, 'training-only conditional path complete', flush=True)
    files = [source, Path(__file__), ROOT/'src/certified_frfs/conditional_certified_backward.py',
             ROOT/'src/certified_frfs/joint_certified_backward.py', ROOT/'research/conditional-backward-screen-protocol.md',
             ROOT/'data/processed'/(name+'.npz')]
    (FOLDER/(name+'.json')).write_text(json.dumps({'dataset': name, 'd': X.shape[1], 'references': records,
        'fingerprints': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            future.result()
    rows = []
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text())
        adaptive, static, ratios = [], [], []
        for record in saved['references']:
            if not record['joint_feasible']:
                adaptive.append(0.)
                static.append(0.)
                continue
            adaptive.append(1-len(record['adaptive']['trace'][-1]['features'])/saved['d'])
            static.append(1-len(record['static']['trace'][-1]['features'])/saved['d'])
            common = [r for r in record['same_count_losses'] if r['size'] < saved['d']]
            if common:
                r = min(common, key=lambda r: r['size'])
                ratios.append((r['static_loss']-r['adaptive_loss'])/max(r['static_loss'], 1e-12))
        rows.append({'dataset': name, 'adaptive_removed': float(np.mean(adaptive)),
                     'static_removed': float(np.mean(static)), 'same_count_relative_loss_improvement':
                     float(np.mean(ratios)) if ratios else None})
    valid = [r['same_count_relative_loss_improvement'] for r in rows if r['same_count_relative_loss_improvement'] is not None]
    compression_gain = float(np.mean([r['adaptive_removed']-r['static_removed'] for r in rows]))
    loss_gain = float(np.mean(valid)) if valid else 0.
    report = {'rows': rows, 'mean_adaptive_removed': float(np.mean([r['adaptive_removed'] for r in rows])),
              'mean_static_removed': float(np.mean([r['static_removed'] for r in rows])),
              'mean_compression_gain_pp': 100*compression_gain,
              'same_count_relative_reconstruction_improvement': loss_gain,
              'datasets_with_common_nonfull_size': len(valid),
              'mechanism_gate_passed': loss_gain >= .1 and compression_gain >= -.02,
              'scope': 'all15_all44_reference_only_conditional_reconstruction_not_prediction_or_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print({k: v for k, v in report.items() if k != 'rows'}, flush=True)
