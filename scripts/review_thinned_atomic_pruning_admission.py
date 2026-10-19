"""Whole-cohort training comparison; retain increased-cost/lower-BA cases."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from run_thinned_atomic_pruning_admission import ROOT,FOLDER,digest
from continue_soft_topk_knn_pilot import verify


def main():
    audit=verify(FOLDER);oldfolder=ROOT/'results/atomic-pruning-admission';oldaudit=verify(oldfolder)
    current=json.loads((FOLDER/'complete.json').read_text());old=json.loads((oldfolder/'complete.json').read_text())
    prior={r['dataset']:r for r in old['rows']};assert len(prior)==15 and len(current['rows'])==15
    rows=[];kinds=Counter();reasons=Counter()
    for row in current['rows']:
        saved=json.loads((FOLDER/(row['dataset']+'.json')).read_text());fit=saved['fit']
        for point in fit['checks']:
            kinds[point['kind']+'_checked']+=1
            reasons['global_dependency']+=point['certificate']['dependency_certificate']['upper_bound']>.1
            reasons['classwise_dependency']+=not point['class_certificate']['feasible_all_classes']
            reasons['majority']+=not point['certificate']['majority_feasible']
            reasons['hard_training']+=not point['nonworsening_training_BA']
            reasons['eligible_but_thinning_worse']+=bool(point['accepted'] and point['nonworsening_training_BA'] and not point['thinning_nonworsening'])
        for i in fit['chosen_ids']:kinds[fit['checks'][i]['kind']+'_chosen']+=1
        previous=prior[row['dataset']]
        rows.append({**row,'old_final_count':previous['final_count'],
            'old_final_training_BA':previous['final_training_BA'],
            'additional_removed':previous['final_count']-row['final_count'],
            'new_minus_old_training_pp':100*(row['final_training_BA']-previous['final_training_BA'])})
    result={'rows':rows,'mean_removed':current['mean_removed'],
        'old_mean_removed':old['mean_removed'],
        'dataset_balanced_training_thinning_gain_pp':float(np.mean([100*(r['final_thinning_BA']-r['initial_thinning_BA']) for r in rows])),
        'datasets_further_reduced':sum(r['additional_removed']>0 for r in rows),
        'datasets_less_reduced':sum(r['additional_removed']<0 for r in rows),
        'dataset_balanced_new_minus_old_training_pp':float(np.mean([r['new_minus_old_training_pp'] for r in rows])),
        'proposal_counts':dict(kinds),'overlapping_rejection_counts':dict(reasons),
        'half_support_targets_reached':sum(r['status']=='target_reached' for r in rows),
        'audit_sha256':digest(FOLDER/'complete-audit.json'),'old_audit_sha256':digest(oldfolder/'complete-audit.json'),
        'review_script_sha256':digest(Path(__file__)),'goal_achieved':False,
        'limits':['Complete training paths from exposed source choices, no query predictions.',
            'Same48-check cap and rescale/atomic donor pools; paths can diverge after selecting different endpoints.',
            'Thinning uses all training labels, so it is a training objective rather than cross-validation.',
            'Each path is nonworsening relative to its own current parent; superiority over a different path is not guaranteed.',
            'No irreducibility, optimum, population accuracy or scientific significance claim.']}
    (ROOT/'research/thinned-atomic-pruning-admission-review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
