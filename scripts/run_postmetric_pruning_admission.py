"""ALL15 audited deployed metrics, new training-only pruning feasibility."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT,FOLDER as PARENT,digest
from continue_soft_topk_knn_pilot import verify
from guarded_postmetric_pruning import fit_postmetric_pruning
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit

FOLDER=ROOT/'results/postmetric-pruning-admission'


def run(name):
    files=[Path(__file__),ROOT/'scripts/guarded_postmetric_pruning.py',ROOT/'scripts/hardaware_guarded_soft5_descent.py',
           ROOT/'scripts/audit_postmetric_pruning_admission.py',ROOT/'research/postmetric-pruning-admission-protocol.md',
           PARENT/(name+'.json'),PARENT/'complete-audit.json',ROOT/'data/processed'/(name+'.npz')]
    files+=sorted((ROOT/'src/certified_frfs').glob('*.py'))
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in files}; out=FOLDER/(name+'.json')
    if out.exists():
        saved=json.loads(out.read_text()); assert saved['fingerprints']==hashes
        if saved['complete']: return saved
    pa=verify(PARENT); assert pa['artifact_hashes'][name]==digest(PARENT/(name+'.json'))
    parent=json.loads((PARENT/(name+'.json')).read_text())
    for rel,sha in parent['fingerprints'].items(): assert digest(ROOT/rel)==sha
    metric=parent['outer']['results']['proposed']['metric']
    X,y=load_dataset(name); ri=np.array(parent['reference_indices']); A,_=preprocess_fit(X[ri],X[ri][:1])
    fs,w,fit=fit_postmetric_pruning(A,y[ri],metric['features'],metric['weights'],fraction=.5,pool=4,max_checks=48)
    assert fit['guard_reference']=='original_full' and fit['endpoint_guard']['accepted']
    assert hashes=={p.relative_to(ROOT).as_posix():digest(p) for p in files}
    saved={'dataset':name,'d':X.shape[1],'reference_indices':ri.tolist(),'initial_features':metric['features'],
        'initial_weights':metric['weights'],'features':fs,'weights':w.tolist(),'fit':fit,'fingerprints':hashes,
        'complete':True,'query_labels_or_accuracy_used':False,'goal_achieved':False,
        'scope':'all15_training_only_pruning_from_audited_deployed_weights_not_prediction'}
    out.write_text(json.dumps(saved,separators=(',',':')),encoding='utf-8')
    print(name,'initial',len(metric['features']),'final',len(fs),'checks',len(fit['checks']),fit['status'],
          'hard_training_gain_pp',100*(fit['final_hard_LOO_BA']-fit['initial_hard_LOO_BA']),flush=True)
    return saved


def main():
    threadpool_limits(1); FOLDER.mkdir(exist_ok=True); names=sorted(verify(PARENT)['artifact_hashes'])
    reports=[run(name) for name in names]
    rows=[{'dataset':r['dataset'],'d':r['d'],'initial_count':len(r['initial_features']),
        'final_count':len(r['features']),'status':r['fit']['status'],'checks':len(r['fit']['checks']),
        'initial_hard_LOO_BA':r['fit']['initial_hard_LOO_BA'],'final_hard_LOO_BA':r['fit']['final_hard_LOO_BA']} for r in reports]
    result={'rows':rows,'datasets':len(rows),'newly_pruned_datasets':sum(r['final_count']<r['initial_count'] for r in rows),
        'mean_initial_removed':float(np.mean([1-r['initial_count']/r['d'] for r in rows])),
        'mean_final_removed':float(np.mean([1-r['final_count']/r['d'] for r in rows])),
        'dataset_balanced_hard_training_gain_pp':float(np.mean([100*(r['final_hard_LOO_BA']-r['initial_hard_LOO_BA']) for r in rows])),
        'goal_achieved':False,'independent_audit_required':True,
        'scope':'all15_training_compression_feasibility_not_test_gain_or_fair_recent_method_comparison'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)


if __name__=='__main__': main()
