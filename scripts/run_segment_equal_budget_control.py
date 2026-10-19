"""ALL86 training reference controls, label-independent fixed16 grid."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_guarded_hard5_segment_admission import ROOT,FOLDER as SEGMENT,PARENT,digest
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from audit_flexible_majority_repair_admission import independent_protected
from audit_classwise_repair_admission import independent_class_certificate
from audit_simplex_source_budget_diagnostic import independent_loo
from hardaware_guarded_soft5_descent import hard5_LOO_BA

FOLDER=ROOT/'results/segment-equal-budget-control'


def fixed_grid():
    grid=[0.,.05,.1,.25,.5,.75,1.]
    while len(grid)<16:
        ordered=sorted(grid); pairs=list(zip(ordered[:-1],ordered[1:]))
        lo,hi=min(pairs,key=lambda pair:(-(pair[1]-pair[0]),pair[0]))
        grid.append(lo+(hi-lo)/2)
    return tuple(grid)


def run(name):
    path=FOLDER/(name+'.json'); sources=[Path(__file__),ROOT/'research/hard5-segment-equal-budget-control-protocol.md',
        SEGMENT/(name+'.json'),SEGMENT/'complete-audit.json',PARENT/(name+'.json'),PARENT/'complete-audit.json',
        ROOT/'scripts/audit_simplex_source_budget_diagnostic.py',ROOT/'scripts/hardaware_guarded_soft5_descent.py',
        ROOT/'scripts/audit_flexible_majority_repair_admission.py',ROOT/'scripts/audit_classwise_repair_admission.py',
        ROOT/'data/processed'/(name+'.npz')]
    sources+=sorted((ROOT/'src/certified_frfs').glob('*.py'))
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in sources}
    if path.exists():
        stored=json.loads(path.read_text()); assert stored['fingerprints']==hashes
        if stored['complete']: return stored
    sa=verify(SEGMENT); pa=verify(PARENT)
    assert sa['artifact_hashes'][name]==digest(SEGMENT/(name+'.json'))
    assert pa['artifact_hashes'][name]==digest(PARENT/(name+'.json'))
    proposal=json.loads((SEGMENT/(name+'.json')).read_text()); parent=json.loads((PARENT/(name+'.json')).read_text())
    assert proposal['reference_indices']==parent['reference_indices'] and len(proposal['records'])==len(parent['records'])
    for rel,sha in proposal['fingerprints'].items(): assert digest(ROOT/rel)==sha
    X,y=load_dataset(name); ri=np.array(parent['reference_indices']); labels=y[ri]
    A,_=preprocess_fit(X[ri],X[ri][:1]); mass=np.sqrt(A.shape[1]); full=np.ones(A.shape[1])/mass
    protected=independent_protected(A,labels,full); records=[]
    for adaptive,old in zip(proposal['records'],parent['records'],strict=True):
        fs=old['features']; assert fs==adaptive['features']
        anchor=np.array(old['anchor']); target=np.array(old['target']); checks=[]
        for t in fixed_grid():
            w=(1-t)*anchor+t*target
            cert=weight_certificate(A,labels,fs,full,w,protected=protected)
            classes=independent_class_certificate(labels,cert)
            ba=hard5_LOO_BA(A[:,fs],labels,w); scalar=independent_loo(A,labels,fs,w)
            assert abs(ba-scalar)<=1e-12
            checks.append({'t':t,'weights':w.tolist(),'hard_full_training_LOO_BA':ba,
                'certificate':cert,'class_certificate':classes,
                'accepted':bool(cert['feasible'] and classes['feasible_all_classes'])})
        admitted=[i for i,p in enumerate(checks) if p['accepted']]
        assert checks[0]['accepted'] and checks[6]['accepted']
        chosen=min(admitted,key=lambda i:(-round(checks[i]['hard_full_training_LOO_BA'],12),abs(1-checks[i]['t']),i))
        fixed_ba=checks[chosen]['hard_full_training_LOO_BA']; segment_ba=adaptive['fit']['final_hard_LOO_BA']
        records.append({'features':fs,'anchor':anchor.tolist(),'input_target':target.tolist(),
            'checks':checks,'selected':chosen,'fixed_best_BA':fixed_ba,'segment_best_BA':segment_ba,
            'segment_candidate_count':len(adaptive['fit']['checks']),'segment_minus_fixed_pp':100*(segment_ba-fixed_ba)})
        print(name,'support',len(records),'segment_minus_equal16_pp',records[-1]['segment_minus_fixed_pp'],flush=True)
    assert hashes=={p.relative_to(ROOT).as_posix():digest(p) for p in sources}
    result={'dataset':name,'reference_indices':ri.tolist(),'fingerprints':hashes,'records':records,
        'grid':fixed_grid(),'complete':True,'query_labels_or_accuracy_used':False,'goal_achieved':False,
        'independent_scalar_vote_replay_passed':True,
        'scope':'all86_training_finite_controls_not_classifier_test_or_exhaustive_certified_search'}
    path.write_text(json.dumps(result,separators=(',',':')),encoding='utf-8'); return result


def main():
    threadpool_limits(1); FOLDER.mkdir(exist_ok=True); names=sorted(verify(SEGMENT)['artifact_hashes'])
    reports=[run(name) for name in names]; rows=[]
    for report in reports:
        gains=[r['segment_minus_fixed_pp'] for r in report['records']]
        rows.append({'dataset':report['dataset'],'supports':len(gains),'mean_segment_minus_fixed_pp':float(np.mean(gains)),
            'segment_wins':sum(g>1e-9 for g in gains),'segment_losses':sum(g < -1e-9 for g in gains),
            'ties':sum(abs(g)<=1e-9 for g in gains),
            'segment_checks':sum(r['segment_candidate_count'] for r in report['records']),
            'fixed_checks':16*len(gains)})
    assert len(rows)==15 and sum(r['supports'] for r in rows)==86
    result={'rows':rows,'datasets':15,'supports':86,'grid':fixed_grid(),
        'dataset_balanced_segment_minus_fixed_pp':float(np.mean([r['mean_segment_minus_fixed_pp'] for r in rows])),
        'independent_scalar_vote_replay_passed':True,'goal_achieved':False,
        'scope':'equal_max16_training_budget_control_not_inferential_or_predictive_advantage'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)


if __name__=='__main__': main()
