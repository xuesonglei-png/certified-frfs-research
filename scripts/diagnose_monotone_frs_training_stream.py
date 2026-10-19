"""Read-only mechanism diagnosis of raw Fair28 without loading whole cohorts.

This is provisional until the independent classifier audit finishes. No new
fits, label selection, metric changes, statistical significance or promotion.
"""
import json
import time
from collections import Counter
import ijson
import numpy as np
from run_simplex_soft5_knn_pilot import ROOT,digest

FOLDER=ROOT/'results/monotone-frs-inner'
TARGET=ROOT/'research/monotone-frs-training-mechanism-provisional.json'


def main():
    summary_path=FOLDER/'complete.json';summary=json.loads(summary_path.read_text())
    assert summary['datasets']==15 and summary['inner_folds']==44
    summary_sha=digest(summary_path);rows=[];aggregate=Counter()
    for row in summary['rows']:
        name=row['dataset'];path=FOLDER/(name+'.json')
        assert summary['artifact_hashes'][name]==digest(path)
        with path.open('rb') as stream:
            selection=next(ijson.items(stream,'selection',use_float=True))
        counts=Counter();trajectories=[]
        with path.open('rb') as stream:
            for learner in ijson.items(stream,'inner.item.stage.learners.item',use_float=True):
                if learner['family']!='one_sided_cell_constrained':
                    continue
                arm='proposed' if learner['certified_arm'] else 'controls'
                fit=learner['fit'];counts[arm+'_fits']+=1
                counts[arm+'_unavailable_cells']+=fit['cell_unavailable']
                if fit['cell_unavailable']:
                    continue
                l1=float(np.sum(np.abs(np.asarray(learner['target'])-np.asarray(learner['anchor'])))/fit['mass'])
                gain=fit['training_objective_initial']-fit['training_objective_final']
                bounds=fit['class_cell_final']['raw_upper_bounds_with_inflation']
                counts[arm+'_strict_training_descent']+=gain>1e-10
                counts[arm+'_nonzero_metric_change']+=l1>1e-10
                counts[arm+'_active_class_bounds']+=sum(abs(v-.1)<1e-7 for v in bounds)
                counts[arm+'_class_bounds']+=len(bounds)
                counts[arm+'_subproblem_failures']+=fit['subproblem_failures']
                trajectories.append({'arm':arm,'training_objective_decrease':float(gain),
                    'normalized_weight_L1_change':l1,'class_upper_bounds':bounds,
                    'reference_scope':fit['reference_scope'],'exact_zero_weights':sum(w==0. for w in learner['target'])})
        assert summary['artifact_hashes'][name]==digest(path)
        aggregate.update(counts)
        rows.append({'dataset':name,'source_sha256':summary['artifact_hashes'][name],
            'selected_candidate_indices':{m:r['selected'] for m,r in selection.items()},
            'new_family_selected':{m:r['selected']>=21 for m,r in selection.items()},
            'proposed_validation_gain_vs_previous_pp':row['vs_previous_DC_pp']['proposed'],
            'counts':dict(counts),'training_trajectories':trajectories})
        print(name,'raw read-only one-sided mechanism',dict(counts),flush=True)
    assert digest(summary_path)==summary_sha
    report={'created_unix':time.time(),'source_summary_sha256':summary_sha,
        'all_dataset_rows':rows,'aggregate_counts':dict(aggregate),
        'classification_audit_complete_at_end':(FOLDER/'complete-audit.json').exists(),
        'provisional':True,'goal_achieved':False,'classification_superiority_established':False,
        'new_fits':0,'new_predictions':0,
        'limits':['Mechanism diagnosis of raw results pending full independent audit.',
                  'Training descent and metric movement do not establish improved prediction.',
                  'All15 exposed datasets retained; no data-based hyperparameter change.']}
    assert not TARGET.exists()
    TARGET.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
