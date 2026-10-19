"""Supplement audited training review with capped-bound alignment and zeros."""
import json
from collections import Counter
import numpy as np
from run_frs_one_sided_training import ROOT, FOLDER, digest
from continue_soft_topk_knn_pilot import verify


def main():
    audit = verify(FOLDER); rows = []; counts = Counter(); fold_rows = []
    summary = json.loads((FOLDER/'complete.json').read_text())
    for sr in summary['rows']:
        name = sr['dataset']; saved = json.loads((FOLDER/(name+'.json')).read_text()); local = []
        for fi,b in enumerate(saved['inner_training']):
            if not b['strict_cell_available']:
                counts['unavailable_cells'] += 1; continue
            old,new = b['fits']['symmetric'],b['fits']['one_sided']; a,z = old['fit'],new['fit']
            gain = a['training_objective_final']-z['training_objective_final']
            boundgain = 100*(a['training_error_upper_bound_final']-z['training_error_upper_bound_final'])
            row = {'dataset':name,'fold':fi,'training_objective_decrease':gain,'clipped_training_bound_decrease_pp':boundgain,
                'target_outside_symmetric_cell':bool(new['one_sided_bound_at_target']['feasible'] and not new['symmetric_bound_at_target']['feasible']),
                'target_exact_zero_weights':int(sum(w==0 for w in new['target'])),
                'objective_improves_but_clipped_bound_worsens':bool(gain>1e-9 and boundgain < -1e-8)}
            local.append(row);fold_rows.append(row); counts['folds']+=1
            counts['objective_better_folds']+=gain>1e-9;counts['objective_worse_folds']+=gain < -1e-9
            counts['clipped_bound_better_folds']+=boundgain>1e-8;counts['clipped_bound_worse_folds']+=boundgain < -1e-8
            counts['objective_improves_but_bound_worsens']+=row['objective_improves_but_clipped_bound_worsens']
            counts['target_expansion_folds']+=row['target_outside_symmetric_cell']
            counts['folds_with_exact_zero_weight']+=row['target_exact_zero_weights']>0
            counts['optimizer_failures']+=a['subproblem_failures']+z['subproblem_failures']
        rows.append({'dataset':name,'folds':len(local),
            'mean_training_objective_decrease':float(np.mean([r['training_objective_decrease'] for r in local])) if local else None,
            'mean_clipped_training_bound_decrease_pp':float(np.mean([r['clipped_training_bound_decrease_pp'] for r in local])) if local else None})
    assert len(rows)==15 and counts['folds']+counts['unavailable_cells']==44
    out={'audit_sha256':digest(FOLDER/'complete-audit.json'),'all_dataset_rows':rows,'all_fold_rows':fold_rows,'counts':dict(counts),
        'dataset_balanced_objective_decrease':float(np.mean([r['mean_training_objective_decrease'] for r in rows if r['folds']])),
        'dataset_balanced_clipped_training_bound_decrease_pp':float(np.mean([r['mean_clipped_training_bound_decrease_pp'] for r in rows if r['folds']])),
        'goal_achieved':False,'classification_scored':False,'classification_superiority_established':False,
        'decision':'Training feasible-region expansion is real; prospectively test monotone family equally in28choice inner diagnostic. Capped objective needs its separate training audit.',
        'limits':['No extra exact-zero attributes were established unless recorded.','Training loss and finite-table error bounds are not test accuracy.',
                  'All15 exposed; no significance claim or outer promotion.']}
    (ROOT/'research/one-sided-FRS-alignment-review.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('all_dataset_rows','all_fold_rows')},indent=2),flush=True)


if __name__=='__main__':
    main()
