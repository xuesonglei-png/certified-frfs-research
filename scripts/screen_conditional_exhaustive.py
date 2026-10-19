"""Remove shortlist truncation, with unchanged adaptive costs/certificates."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import screen_conditional_backward as original
from certified_frfs.conditional_certified_backward import conditional_backward_path

original.FOLDER = original.ROOT/'results/conditional-exhaustive-screen'


def exhaustive_path(X, y, coef):
    return conditional_backward_path(X, y, coef, checks=X.shape[1])


original.conditional_backward_path = exhaustive_path


def run(name):
    original.run(name)
    path = original.FOLDER/(name+'.json'); saved = json.loads(path.read_text())
    for f in (Path(__file__), original.ROOT/'research/conditional-exhaustive-screen-protocol.md'):
        saved['fingerprints'][str(f.relative_to(original.ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
    path.write_text(json.dumps(saved, indent=2), encoding='utf-8')


if __name__ == '__main__':
    original.FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((original.SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            future.result()
    rows = []
    for name in names:
        saved = json.loads((original.FOLDER/(name+'.json')).read_text())
        adaptive, static, ratios = [], [], []
        for record in saved['references']:
            if not record['joint_feasible']:
                adaptive.append(0.); static.append(0.)
                continue
            adaptive.append(1-len(record['adaptive']['trace'][-1]['features'])/saved['d'])
            static.append(1-len(record['static']['trace'][-1]['features'])/saved['d'])
            common = [r for r in record['same_count_losses'] if r['size'] < saved['d']]
            if common:
                r = min(common, key=lambda r: r['size'])
                ratios.append((r['static_loss']-r['adaptive_loss'])/max(r['static_loss'], 1e-12))
        rows.append({'dataset': name, 'adaptive_removed': float(np.mean(adaptive)),
                     'static_removed': float(np.mean(static)), 'same_count_relative_loss_improvement': float(np.mean(ratios)) if ratios else None})
    valid = [r['same_count_relative_loss_improvement'] for r in rows if r['same_count_relative_loss_improvement'] is not None]
    compression = float(np.mean([r['adaptive_removed']-r['static_removed'] for r in rows]))
    loss = float(np.mean(valid)) if valid else 0.
    report = {'rows': rows, 'mean_adaptive_removed': float(np.mean([r['adaptive_removed'] for r in rows])),
              'mean_static_removed': float(np.mean([r['static_removed'] for r in rows])),
              'mean_compression_gain_pp': 100*compression, 'same_count_relative_reconstruction_improvement': loss,
              'datasets_with_common_nonfull_size': len(valid), 'mechanism_gate_passed': loss >= .1 and compression >= -.02,
              'scope': 'complete_conditional_feasible_deletion_search_reference_only_no_prediction_or_global_optimality'}
    (original.FOLDER/'complete.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print({k: v for k, v in report.items() if k != 'rows'}, flush=True)
