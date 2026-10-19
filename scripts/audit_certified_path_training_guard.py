"""Independent scalar predictions, original indices and classwise path choice."""
import numpy as np
from independent_crossgroup_soft5 import independent_groups
from independent_geometric_evaluation import independent_evaluate
from audit_certified_metric_knn_pool import query_certificate
from audit_forest_compact_transfer import equivalent

ARMS=('nominal_class_guard','robust_class_guard')


def check_guard(A,y,joint,saved,counts):
    expected=[{'source_step':-1,'features':joint['original_support'],'weights':joint['original_anchor'],'kind':'original_anchor'}]
    if joint['available']:
        for i,step in enumerate(joint['eliminations']['pair_transition']['steps']):
            if step['accepted']:
                expected.append({'source_step':i,'features':joint['original_support'],'weights':step['target'],'kind':'independently_audited_joint_face'})
    equivalent(expected,saved['candidates']);assert len(expected)==len(saved['training_reports'])
    group=independent_groups(y);classes=np.unique(y);reports=[];metrics=[]
    for candidate,stored in zip(expected,saved['training_reports'],strict=True):
        w,fs=candidate['weights'],candidate['features'];keep=[j for j,v in enumerate(w) if v!=0.]
        metric={'features':[fs[j] for j in keep],'weights':[w[j] for j in keep]};metrics.append(metric)
        pred=np.empty_like(y);flags=np.zeros(len(y),bool);blocks=[]
        for g in sorted(set(group.tolist())):
            refs=np.flatnonzero(group!=g);queries=np.flatnonzero(group==g);assert len(refs)>=5
            r=independent_evaluate(A[refs],y[refs],A[queries],y[queries],metric)
            prediction=np.asarray(r['predictions']);pred[queries]=prediction
            cert=query_certificate(A[refs],y[refs],A[queries],prediction,metric['features'],metric['weights']);flags[queries]=cert['certified']
            blocks.append({'group':g,'references':refs.tolist(),'queries':queries.tolist(),'predictions':prediction.tolist(),
                'certified':cert['certified'],'neighbor_indices':[[int(refs[j]) for j in row] for row in r['neighbor_indices']]})
            counts['independent_training_crossgroup_predictions']+=1
        correct=pred==y;ordinary=[int(sum(correct[y==c])) for c in classes];robust=[int(sum((correct&flags)[y==c])) for c in classes]
        totals=[int(sum(y==c)) for c in classes]
        report={'groups':group.tolist(),'blocks':blocks,'classes':classes.tolist(),'class_counts':totals,
            'class_correct_counts':ordinary,'class_certified_correct_counts':robust,
            'balanced_accuracy':float(np.mean([a/b for a,b in zip(ordinary,totals,strict=True)])),
            'certified_balanced_accuracy':float(np.mean([a/b for a,b in zip(robust,totals,strict=True)])),
            'query_count':len(y),'scope':'same_training_fold_crossgroup_hard_vote_not_independent_validation'}
        equivalent(report,stored);reports.append(report)
    choices={};anchor=np.asarray(joint['original_anchor'])
    for arm in ARMS:
        eligible=[]
        for i,r in enumerate(reports):
            if all(a>=b for a,b in zip(r['class_correct_counts'],reports[0]['class_correct_counts'],strict=True)) and (
                arm=='nominal_class_guard' or all(a>=b for a,b in zip(r['class_certified_correct_counts'],reports[0]['class_certified_correct_counts'],strict=True))):eligible.append(i)
        selected=min(eligible,key=lambda i:(len(metrics[i]['features']),-reports[i]['balanced_accuracy'],
            -reports[i]['certified_balanced_accuracy'],sum(abs(np.asarray(expected[i]['weights'])-anchor)),i))
        choice={'arm':arm,'eligible_indices':eligible,'selected_index':selected,'training_per_class_correct_counts_nondecreasing':True,
            'training_per_class_certified_correct_counts_nondecreasing':arm=='robust_class_guard',
            'new_query_labels_used':False,'validation_accuracy_guaranteed':False}
        equivalent(choice,saved['choices'][arm]);equivalent(metrics[selected],saved['metrics'][arm]);choices[arm]=choice
    assert not saved['query_labels_used_for_selection'] and saved['selection_scope']=='training_only_all_class_nondegradation_on_audited_joint_path'
    counts['independent_training_path_guard_selections']+=2
