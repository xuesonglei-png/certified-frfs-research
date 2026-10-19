"""Separate shared-classifier effects from within-family selector contrasts."""
import hashlib
import json
from pathlib import Path
import numpy as np
from review_box_iterations import paired

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'results/crossfit-label-loss-recent-pilot'
NEW = ROOT/'results/penalized-forest-recent-pilot'


def hashfile(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    old, new = [json.loads((folder/'complete.json').read_text()) for folder in (OLD, NEW)]
    old_rows = {r['dataset']: r for r in old['rows']}
    assert set(old_rows) == {r['dataset'] for r in new['rows']}
    rows, supports = [], []
    for row in new['rows']:
        name = row['dataset']
        prior = old_rows[name]
        # Identical raw outer/inner partitions, model grid and baseline paths.
        a, b = [json.loads((folder/(name+'.json')).read_text()) for folder in (OLD, NEW)]
        for key in ('reference_indices', 'validation_indices', 'raw_n', 'd'):
            assert a[key] == b[key], key
        for x, y in zip(a['inner']+[a['outer']], b['inner']+[b['outer']]):
            assert x['paths'] == y['paths']
            for key in ('reference_indices', 'validation_indices'):
                if key in x:
                    assert x[key] == y[key]
        values = {method: 100*(row['metrics'][method]['clean_BA']-prior['metrics'][method]['clean_BA'])
                  for method in new['means']}
        rows.append({'dataset': name, 'shared_graph_change_BA_pp': values,
                     'new_proposed_minus_matched_FGS_pp': 100*(row['metrics']['proposed']['clean_BA']-row['metrics']['fgs_matched']['clean_BA']),
                     'new_proposed_minus_matched_SFRIP_pp': 100*(row['metrics']['proposed']['clean_BA']-row['metrics']['sfrip_matched']['clean_BA'])})
        for stage_index, (x, y) in enumerate(zip(a['inner']+[a['outer']], b['inner']+[b['outer']])):
            s, t = x['stage'], y['stage']
            supports.append({'dataset': name, 'stage_index': stage_index,
                'old_count': s['actual_count'], 'new_count': t['actual_count'],
                'same_support': s['candidate']['features'] == t['candidate']['features'],
                'old_teacher_edges': len(s['tree']['edges']), 'new_teacher_edges': len(t['tree']['edges']),
                'new_selected_model_config': t['teacher_selection']['options'][t['teacher_selection']['chosen_index']]['config']})
    contrasts = {method: paired([r['shared_graph_change_BA_pp'][method]/100 for r in rows]) for method in new['means']}
    summary = {'summary_sha256': {'old': hashfile(OLD/'complete.json'), 'new': hashfile(NEW/'complete.json')},
               'new_independently_audited': (NEW/'complete-audit.json').exists(),
               'rows': rows, 'stage_supports': supports, 'shared_graph_effect_descriptive': contrasts,
               'within_new_family_selector_contrasts': new['contrasts_clean_BA'],
               'fixed_pilot_gate_passed': new['pilot_gate_passed'], 'goal_achieved': False,
               'scope': 'all15_exposed_development_shared_graph_effect_not_selector_superiority_or_confirmation'}
    target = ROOT/'research/penalized-forest-recent-pilot-review.json'
    target.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print({k: v for k, v in summary.items() if k not in ('rows', 'stage_supports')}, flush=True)
