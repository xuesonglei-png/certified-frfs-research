"""Report every exposed development dataset across preserved ranking pilots."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = {'mse': 'adaptive-multiknot-recent-pilot',
        'training_loss': 'projected-label-loss-recent-pilot',
        'crossfit_loss': 'crossfit-label-loss-recent-pilot'}


def review():
    summaries, artifacts, hashes = {}, {}, {}
    import hashlib
    for arm, folder in ARMS.items():
        path = ROOT/'results'/folder/'complete.json'
        if not path.exists():
            continue
        summaries[arm] = json.loads(path.read_text())
        hashes[arm] = hashlib.sha256(path.read_bytes()).hexdigest()
        artifacts[arm] = {r['dataset']: json.loads((path.parent/(r['dataset']+'.json')).read_text())
                          for r in summaries[arm]['rows']}
    rows = []
    for name in sorted(artifacts['mse']):
        row = {'dataset': name, 'd': artifacts['mse'][name]['d'], 'arms': {}}
        for arm in artifacts:
            saved = artifacts[arm][name]
            stage = saved['outer']['stage']
            blocks = saved['inner']+[saved['outer']]
            attempts = [a for block in blocks for a in block['stage']['path']['attempts']]
            rejected = [a for a in attempts if not a['accepted']]
            srow = next(r for r in summaries[arm]['rows'] if r['dataset'] == name)
            row['arms'][arm] = {'features': stage['candidate']['features'],
                'actual_count': stage['actual_count'], 'target_count': stage['target_count'],
                'mean_outer_metrics': srow['metrics'],
                'target_equals_full': stage['target_count'] == row['d'],
                'all_inner_and_outer_full': all(b['stage']['actual_count'] == row['d'] for b in blocks),
                'outer_stalled_above_target': stage['actual_count'] > stage['target_count'],
                'attempts': len(attempts), 'rejected': len(rejected),
                'majority_failures': sum(not a['certificate']['majority_feasible'] for a in rejected),
                'dependency_failures': sum(a['certificate']['dependency_certificate']['upper_bound'] > .1 for a in rejected)}
        if 'crossfit_loss' in row['arms']:
            mse, oof = row['arms']['mse'], row['arms']['crossfit_loss']
            row['crossfit_minus_mse_BA_pp'] = 100*(oof['mean_outer_metrics']['proposed']['clean_BA']-mse['mean_outer_metrics']['proposed']['clean_BA'])
            row['same_outer_support_mse_crossfit'] = mse['features'] == oof['features']
        rows.append(row)
    result = {'summary_sha256': hashes, 'rows': rows,
              'scope': 'all_exposed_development_datasets_no_confirmatory_selection',
              'aggregate': {}}
    for arm in summaries:
        result['aggregate'][arm] = {key: sum(r['arms'][arm][key] for r in rows)
            for key in ('target_equals_full', 'all_inner_and_outer_full', 'outer_stalled_above_target',
                        'attempts', 'rejected', 'majority_failures', 'dependency_failures')}
        result['aggregate'][arm].update({'means': summaries[arm]['means'],
                                         'gate_passed': summaries[arm]['pilot_gate_passed']})
    if 'crossfit_loss' in summaries:
        result['crossfit_vs_mse'] = {'mean_BA_gain_pp': float(np.mean([r['crossfit_minus_mse_BA_pp'] for r in rows])),
            'changed_supports': sum(not r['same_outer_support_mse_crossfit'] for r in rows)}
    target = ROOT/'research/projected-ranking-sequence-review.json'
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({k: v for k, v in result.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    review()
