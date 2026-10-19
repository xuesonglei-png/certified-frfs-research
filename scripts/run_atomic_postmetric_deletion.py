"""ALL117 fixed parents/donors; receiver deletion is training-only."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT,digest
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.majority_metric import MajorityTerms
from atomic_postmetric_deletion import DistanceProfiles,fit_atomic_deletion

PARENT=ROOT/'results/postmetric-pruning-admission'
FOLDER=ROOT/'results/atomic-postmetric-deletion'


def fingerprints(name):
    files=[Path(__file__),ROOT/'scripts/atomic_postmetric_deletion.py',
        ROOT/'scripts/audit_atomic_postmetric_deletion.py',ROOT/'scripts/hardaware_guarded_soft5_descent.py',
        ROOT/'research/atomic-postmetric-deletion-protocol.md',PARENT/(name+'.json'),
        PARENT/'complete-audit.json',ROOT/'data/processed'/(name+'.npz')]
    files+=sorted((ROOT/'src/certified_frfs').glob('*.py'))
    return {p.relative_to(ROOT).as_posix():digest(p) for p in files}


def run(name,audit):
    path=PARENT/(name+'.json'); assert digest(path)==audit['artifact_hashes'][name]
    source=json.loads(path.read_text()); hashes=fingerprints(name); out=FOLDER/(name+'.json')
    if out.exists():
        saved=json.loads(out.read_text()); assert saved['fingerprints']==hashes
        if saved['complete']: return saved
    X,y=load_dataset(name); ri=np.array(source['reference_indices']); A,_=preprocess_fit(X[ri],X[ri][:1]); labels=y[ri]
    profiles=DistanceProfiles(A); full=np.ones(A.shape[1])/np.sqrt(A.shape[1])
    protected=MajorityTerms(A,labels,weights=full).bounds(range(A.shape[1]),allowed_enemies=2)>1e-10
    records=[]
    for old in source['fit']['checks']:
        fit=fit_atomic_deletion(A,labels,old['features_before'],old['weights_before'],
            old['removed_feature'],old['BA_before'],geometry=profiles,protected=protected)
        records.append({'source_candidate_id':old['id'],'old_eligible':old['eligible'],
            'old_rescaled_BA':old['hard_training_LOO_BA'],'fit':fit})
        print(name,'candidate',old['id'],'old',old['eligible'],'atomic',fit['selected_id'] is not None,flush=True)
    assert hashes==fingerprints(name)
    saved={'dataset':name,'reference_indices':ri.tolist(),'records':records,'fingerprints':hashes,
        'complete':True,'query_labels_used':False,'goal_achieved':False,
        'scope':'fixed117_training_atomic_endpoints_not_new_prediction_pipeline'}
    out.write_text(json.dumps(saved,separators=(',',':')),encoding='utf-8'); return saved


def summary_counts(rows):
    return {'datasets':len(rows),'candidates':sum(len(r['records']) for r in rows),
        'atomic_checks':sum(len(p['fit']['checks']) for r in rows for p in r['records']),
        'old_eligible':sum(p['old_eligible'] for r in rows for p in r['records']),
        'atomic_eligible':sum(p['fit']['selected_id'] is not None for r in rows for p in r['records']),
        'newly_eligible':sum(not p['old_eligible'] and p['fit']['selected_id'] is not None for r in rows for p in r['records']),
        'old_eligible_no_atomic':sum(p['old_eligible'] and p['fit']['selected_id'] is None for r in rows for p in r['records']),
        'datasets_with_new_eligible':sum(any(not p['old_eligible'] and p['fit']['selected_id'] is not None for p in r['records']) for r in rows)}


def main():
    threadpool_limits(1); FOLDER.mkdir(exist_ok=True); audit=verify(PARENT)
    names=sorted(audit['artifact_hashes']); assert len(names)==15
    reports=[run(name,audit) for name in names]; counts=summary_counts(reports); assert counts['candidates']==117
    result={'counts':counts,'independent_audit_required':True,'goal_achieved':False,
        'scope':'all15_fixed_candidate_training_receiver_diagnosis_not_prediction'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)


if __name__=='__main__': main()
