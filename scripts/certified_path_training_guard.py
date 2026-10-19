"""Select an audited certified path endpoint using training-only hard 5NN.

The guard preserves training class counts only; it is not a population or
validation accuracy guarantee. It never changes original FRS certification.
"""
import numpy as np
from crossgroup_soft5_terms import assignments
from geometric_vote_evaluation import evaluate
from certified_frfs.knn_query import certify_knn_queries

ARMS=('nominal_class_guard','robust_class_guard')


def path_candidates(joint):
    fs=joint['original_support'];anchor=joint['original_anchor']
    out=[{'source_step':-1,'features':fs,'weights':anchor,'kind':'original_anchor'}]
    if joint['available']:
        for index,step in enumerate(joint['eliminations']['pair_transition']['steps']):
            if step['accepted']:
                assert step['target'] is not None
                out.append({'source_step':index,'features':fs,'weights':step['target'],'kind':'independently_audited_joint_face'})
    return out


def metric_of(candidate):
    fs,w=candidate['features'],candidate['weights'];keep=[j for j,v in enumerate(w) if v!=0.]
    return {'features':[fs[j] for j in keep],'weights':[w[j] for j in keep]}


def training_report(A,y,candidate):
    A,y=np.asarray(A,float),np.asarray(y);group=assignments(y);metric=metric_of(candidate)
    predictions=np.empty_like(y);certified=np.zeros(len(y),bool);blocks=[]
    for g in sorted(set(group.tolist())):
        refs=np.flatnonzero(group!=g);queries=np.flatnonzero(group==g)
        if len(refs)<5:raise ValueError('At least five cross-group references required; never silently exclude queries')
        report=evaluate(A[refs],y[refs],A[queries],y[queries],metric)
        pred=np.asarray(report['predictions']);predictions[queries]=pred
        if metric['features']:
            embedded=np.zeros(A.shape[1]);embedded[metric['features']]=metric['weights']
            cert=certify_knn_queries(A[refs],y[refs],A[queries],pred,metric['features'],weights=embedded,k=5,allowed_enemies=2)
            flags=np.asarray(cert['certified'],bool)
        else:flags=np.zeros(len(queries),bool)
        certified[queries]=flags
        blocks.append({'group':g,'references':refs.tolist(),'queries':queries.tolist(),'predictions':pred.tolist(),
            'certified':flags.tolist(),'neighbor_indices':[[int(refs[j]) for j in row] for row in report['neighbor_indices']]})
    classes=np.unique(y);correct=predictions==y
    counts=[int(sum(correct[y==c])) for c in classes];cert_counts=[int(sum((correct&certified)[y==c])) for c in classes]
    totals=[int(sum(y==c)) for c in classes]
    return {'groups':group.tolist(),'blocks':blocks,'classes':classes.tolist(),'class_counts':totals,
        'class_correct_counts':counts,'class_certified_correct_counts':cert_counts,
        'balanced_accuracy':float(np.mean(np.asarray(counts)/totals)),
        'certified_balanced_accuracy':float(np.mean(np.asarray(cert_counts)/totals)),
        'query_count':len(y),'scope':'same_training_fold_crossgroup_hard_vote_not_independent_validation'}


def choose(candidates,reports,arm):
    assert arm in ARMS and len(candidates)==len(reports)>0
    base=reports[0];eligible=[];anchor=np.asarray(candidates[0]['weights'])
    for i,(candidate,report) in enumerate(zip(candidates,reports,strict=True)):
        assert report['classes']==base['classes'] and report['class_counts']==base['class_counts']
        ordinary=all(a>=b for a,b in zip(report['class_correct_counts'],base['class_correct_counts'],strict=True))
        robust=all(a>=b for a,b in zip(report['class_certified_correct_counts'],base['class_certified_correct_counts'],strict=True))
        if ordinary and (arm==ARMS[0] or robust):eligible.append(i)
    assert 0 in eligible
    def key(i):
        report=reports[i];w=np.asarray(candidates[i]['weights'])
        return (int(np.count_nonzero(w)),-report['balanced_accuracy'],-report['certified_balanced_accuracy'],
            float(abs(w-anchor).sum()),i)
    selected=min(eligible,key=key)
    return {'arm':arm,'eligible_indices':eligible,'selected_index':selected,
        'training_per_class_correct_counts_nondecreasing':True,
        'training_per_class_certified_correct_counts_nondecreasing':arm==ARMS[1],
        'new_query_labels_used':False,'validation_accuracy_guaranteed':False}


def guard(A,y,joint):
    candidates=path_candidates(joint);reports=[training_report(A,y,c) for c in candidates]
    choices={a:choose(candidates,reports,a) for a in ARMS}
    return {'candidates':candidates,'training_reports':reports,'choices':choices,
        'metrics':{a:metric_of(candidates[choices[a]['selected_index']]) for a in ARMS},
        'query_labels_used_for_selection':False,'selection_scope':'training_only_all_class_nondegradation_on_audited_joint_path'}
