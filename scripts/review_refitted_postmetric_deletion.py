"""Diagnose all fixed deletion/refit candidates after complete replay."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from run_refitted_postmetric_deletion import ROOT,FOLDER,PARENT,digest
from continue_soft_topk_knn_pilot import verify


def candidate_categories(row):
    fit=row['fit']; checks=fit['checks']; initial=checks[0]
    selected=fit['selected_id']
    return {'candidate_count':1,'original_eligible':int(row['original_eligible']),
        'new_eligible':int(selected is not None),
        'newly_eligible':int(not row['original_eligible'] and selected is not None),
        'selected_nonzero_interpolation':int(selected is not None and checks[selected]['interpolation']>0),
        'any_training_nonworsening':int(any(p['nonworsening_parent_BA'] for p in checks)),
        'training_nonworsening_but_no_joint_guard':int(any(p['nonworsening_parent_BA'] for p in checks) and selected is None),
        'none_training_nonworsening':int(not any(p['nonworsening_parent_BA'] for p in checks)),
        'FRS_acceptance_but_no_eligible':int(any(p['accepted'] for p in checks) and selected is None),
        'original_FRS_accepted_hard_harmful':int(initial['accepted'] and not initial['nonworsening_parent_BA']),
        'optimizer_failed':int(not fit['refit']['optimizer_success'])}


def main():
    audit=verify(FOLDER); prior=verify(PARENT); summary=json.loads((FOLDER/'complete.json').read_text())
    assert len(audit['artifact_hashes'])==15 and summary['counts']['candidates']==117
    rows=[]; categories=Counter(); rejections=Counter(); solvers=Counter()
    for name in sorted(audit['artifact_hashes']):
        saved=json.loads((FOLDER/(name+'.json')).read_text()); data=saved['records']
        assert saved['complete'] and not saved['query_labels_used']
        counts=Counter(); points=[]
        for record in data:
            values=candidate_categories(record); counts.update(values); fit=record['fit']
            solvers[fit['refit']['status']]+=1
            for point in fit['checks']:
                rejections['checks']+=1
                rejections['global_dependency']+=point['certificate']['dependency_certificate']['upper_bound']>.1
                rejections['classwise_dependency']+=not point['class_certificate']['feasible_all_classes']
                rejections['majority']+=not point['certificate']['majority_feasible']
                rejections['hard_training']+=not point['nonworsening_parent_BA']
            selected=fit['selected_id']; chosen=None if selected is None else fit['checks'][selected]
            points.append({'source_candidate_id':record['source_candidate_id'],
                'old_eligible':record['original_eligible'],'new_selected_id':selected,
                'selected_interpolation':None if chosen is None else chosen['interpolation'],
                'rescaled_BA':fit['checks'][0]['hard_training_LOO_BA'],
                'selected_BA':None if chosen is None else chosen['hard_training_LOO_BA'],
                'parent_BA':fit['parent_hard_training_LOO_BA'],
                'optimizer_success':fit['refit']['optimizer_success']})
        categories.update(counts)
        rows.append({'dataset':name,'counts':dict(counts),'candidates':points})
    assert categories['candidate_count']==117 and rejections['checks']==468
    assert categories['newly_eligible']==summary['counts']['newly_eligible']
    # t=0 is unchanged and remains a candidate, so no old eligibility is lost.
    assert categories['new_eligible']>=categories['original_eligible']
    result={'rows':rows,'counts':dict(categories),'check_rejection_counts_overlap':dict(rejections),
        'solver_statuses':dict(solvers),
        'datasets_newly_eligible':sum(r['counts'].get('newly_eligible',0)>0 for r in rows),
        'goal_achieved':False,'independent_audit_sha256':digest(FOLDER/'complete-audit.json'),
        'parent_audit_sha256':digest(PARENT/'complete-audit.json'),'review_script_sha256':digest(Path(__file__)),
        'limits':['ALL117 static candidate development diagnosis, no new deletion path or query predictions.',
            'Candidates share training rows and can have different parent supports; no pooled inferential test or mean-risk claim.',
            'Only four interpolations and one soft objective tested; failure does not prove metric infeasibility.',
            'Original dependency bound checker reused during audit; optimizer globality not established.',
            'A training recovery cannot establish predictive superiority or waive fresh confirmation.']}
    (ROOT/'research/refitted-postmetric-deletion-review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__': main()
