"""ALL117 fixed training candidates: rescale versus deletion/refit."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT,digest
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.majority_metric import MajorityTerms
from refit_postmetric_deletion import fit_refitted_deletion

PARENT=ROOT/'results/postmetric-pruning-admission'
FOLDER=ROOT/'results/refitted-postmetric-deletion'


def fingerprints(name):
    files=[Path(__file__),ROOT/'scripts/refit_postmetric_deletion.py',
        ROOT/'scripts/audit_refitted_postmetric_deletion.py',ROOT/'scripts/simplex_soft_topk_targets.py',
        ROOT/'scripts/hardaware_guarded_soft5_descent.py',ROOT/'scripts/soft_topk_metric_objective.py',
        ROOT/'scripts/soft_topk_predictive_metric.py',ROOT/'research/refitted-postmetric-deletion-protocol.md',
        PARENT/(name+'.json'),PARENT/'complete-audit.json',ROOT/'data/processed'/(name+'.npz')]
    files+=sorted((ROOT/'src/certified_frfs').glob('*.py'))
    return {p.relative_to(ROOT).as_posix():digest(p) for p in files}


def run(name,audit):
    path=PARENT/(name+'.json'); assert audit['artifact_hashes'][name]==digest(path)
    prior=json.loads(path.read_text()); hashes=fingerprints(name); out=FOLDER/(name+'.json')
    if out.exists():
        saved=json.loads(out.read_text()); assert saved['fingerprints']==hashes
        if saved['complete']: return saved
    X,y=load_dataset(name); ri=np.array(prior['reference_indices']); A,_=preprocess_fit(X[ri],X[ri][:1]); labels=y[ri]
    full=np.ones(A.shape[1])/np.sqrt(A.shape[1])
    protected=MajorityTerms(A,labels,weights=full).bounds(range(A.shape[1]),allowed_enemies=2)>1e-10
    records=[]
    for point in prior['fit']['checks']:
        record=fit_refitted_deletion(A,labels,point['features'],point['weights'],point['BA_before'],protected=protected)
        records.append({'source_candidate_id':point['id'],'original_eligible':point['eligible'],'fit':record})
        assert record['checks'][0]['eligible']==point['eligible']
        selected=record['selected_id']
        print(name,'candidate',point['id'],'old',point['eligible'],'new',selected is not None,
              'solver',record['refit']['status'],flush=True)
    assert hashes==fingerprints(name)
    saved={'dataset':name,'reference_indices':ri.tolist(),'records':records,'fingerprints':hashes,
        'complete':True,'query_labels_used':False,'goal_achieved':False,
        'scope':'fixed117_training_candidate_diagnostic_not_new_predictive_pipeline'}
    out.write_text(json.dumps(saved,separators=(',',':')),encoding='utf-8'); return saved


def main():
    threadpool_limits(1); FOLDER.mkdir(exist_ok=True); audit=verify(PARENT)
    names=sorted(audit['artifact_hashes']); assert len(names)==15
    rows=[run(name,audit) for name in names]
    total=sum(len(r['records']) for r in rows); assert total==117
    counts={'datasets':15,'candidates':total,'checks':4*total,
        'old_eligible':sum(p['original_eligible'] for r in rows for p in r['records']),
        'new_eligible':sum(p['fit']['selected_id'] is not None for r in rows for p in r['records']),
        'newly_eligible':sum(not p['original_eligible'] and p['fit']['selected_id'] is not None for r in rows for p in r['records']),
        'optimizer_failures':sum(not p['fit']['refit']['optimizer_success'] for r in rows for p in r['records'])}
    summary={'counts':counts,'goal_achieved':False,'independent_audit_required':True,
        'scope':'retrospective_fixed_training_candidates_not_prediction_or_unbiased_generalization'}
    (FOLDER/'complete.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(summary,flush=True)


if __name__=='__main__': main()
