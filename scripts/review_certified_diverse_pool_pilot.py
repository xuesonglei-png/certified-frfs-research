"""Complete-only development review; never choose a variant from query scores."""
import json
from pathlib import Path
import numpy as np
from run_certified_diverse_pool_pilot import ROOT, FOLDER, METHODS, digest


def main():
    summary = json.loads((FOLDER/'complete.json').read_text())
    audit = json.loads((FOLDER/'complete-audit.json').read_text())
    assert audit['passed'] and audit['summary_sha256'] == digest(FOLDER/'complete.json')
    rows = []
    for row in summary['rows']:
        name = row['dataset']; artifact = FOLDER/(name+'.json')
        assert audit['artifact_hashes'][name] == digest(artifact)
        saved = json.loads(artifact.read_text())
        sel = saved['selection']; proposed = sel['options']['proposed'][sel['selected']['proposed']]
        raw = sel['options']['unguarded_pool'][sel['selected']['unguarded_pool']]
        ba = {m: row['metrics'][m]['clean_BA'] for m in METHODS}
        accepted = {tuple(r['features']): r['accepted'] for r in saved['outer']['pool']}
        raw_fs = saved['outer']['results']['unguarded_pool']['features']
        old = json.loads((ROOT/'results/safeguarded-crossfit-recent-pilot'/(name+'.json')).read_text())
        old_ba = old['outer']['results']['proposed']['evaluation']['metrics']['clean_BA']
        rows.append({'dataset': name, 'd': row['d'], 'actual_count': row['actual_counts']['proposed'],
                     'selected_certified_origin': saved['outer']['recipes'][proposed['selector_index']]['origin'],
                     'selected_raw_origin': saved['outer']['recipes'][raw['selector_index']]['origin'],
                     'raw_outer_support_admitted': accepted[tuple(raw_fs)],
                     'certified_inner_BA': proposed['mean_clean_BA'], 'raw_inner_BA': raw['mean_clean_BA'],
                     'certified_outer_BA': ba['proposed'], 'raw_outer_BA': ba['unguarded_pool'],
                     'gain_over_raw_pp': 100*(ba['proposed']-ba['unguarded_pool']),
                     'gain_over_old_guarded_pp': 100*(ba['proposed']-old_ba),
                     'gain_over_matched_FGS_pp': 100*(ba['proposed']-ba['fgs_matched']),
                     'gain_over_matched_SFRIP_pp': 100*(ba['proposed']-ba['sfrip_matched']),
                     'mean_inner_count': proposed['mean_actual_count'],
                     'inner_fallback_folds': sum(len(b['options']['proposed'][sel['selected']['proposed']]['folds'][0]['features']) == row['d'] for b in saved['inner']),
                     'outer_full': row['actual_counts']['proposed'] == row['d']})
    review = {'rows': rows, 'pilot_gate_passed': summary['pilot_gate_passed'], 'goal_achieved': False,
              'mean_gain_over_old_guarded_pp': float(np.mean([r['gain_over_old_guarded_pp'] for r in rows])),
              'full_outer_datasets': [r['dataset'] for r in rows if r['outer_full']],
              'raw_selected_rejected_datasets': [r['dataset'] for r in rows if not r['raw_outer_support_admitted']],
              'limitations': ['Repeatedly exposed development data; descriptive p-values are not confirmation.',
                              'Pool search has equal recipe/model budget; actual distinct certified supports may be fewer.',
                              'FRS certificates do not imply downstream classifier risk or accuracy.',
                              'FD+/FCE+ fidelity and fresh confirmation remain outstanding.'],
              'summary_sha256': digest(FOLDER/'complete.json'), 'audit_sha256': digest(FOLDER/'complete-audit.json')}
    (ROOT/'research/certified-diverse-pool-pilot-review.json').write_text(json.dumps(review, indent=2), encoding='utf-8')
    print(review, flush=True)


if __name__ == '__main__':
    main()
