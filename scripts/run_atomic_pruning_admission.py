"""ALL15 complete atomic/rescale training paths from audited cap1 metrics."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT,FOLDER as SOURCE,digest
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from guarded_atomic_postmetric_pruning import fit_atomic_pruning

FOLDER=ROOT/'results/atomic-pruning-admission'


def fingerprints(name):
    files=[Path(__file__),ROOT/'scripts/guarded_atomic_postmetric_pruning.py',ROOT/'scripts/atomic_postmetric_deletion.py',
        ROOT/'scripts/independent_atomic_pruning.py',ROOT/'scripts/audit_atomic_pruning_admission.py',
        ROOT/'scripts/hardaware_guarded_soft5_descent.py',ROOT/'research/atomic-pruning-admission-protocol.md',
        SOURCE/(name+'.json'),SOURCE/'complete-audit.json',ROOT/'results/atomic-postmetric-deletion/complete-audit.json',
        ROOT/'data/processed'/(name+'.npz')]
    files+=sorted((ROOT/'src/certified_frfs').glob('*.py'))
    return {p.relative_to(ROOT).as_posix():digest(p) for p in files}


def run(name,audit):
    path=SOURCE/(name+'.json');assert digest(path)==audit['artifact_hashes'][name]
    source=json.loads(path.read_text());hashes=fingerprints(name);out=FOLDER/(name+'.json')
    if out.exists():
        saved=json.loads(out.read_text());assert saved['fingerprints']==hashes
        if saved['complete']:return saved
    metric=source['outer']['results']['proposed']['metric'];X,y=load_dataset(name);ri=np.array(source['reference_indices'])
    A,_=preprocess_fit(X[ri],X[ri][:1]);fs,w,fit=fit_atomic_pruning(A,y[ri],metric['features'],metric['weights'])
    assert fit['guard_reference']=='original_full' and fit['endpoint_guard']['accepted']
    assert hashes==fingerprints(name)
    saved={'dataset':name,'d':X.shape[1],'reference_indices':ri.tolist(),'fit':fit,'features':fs,'weights':w.tolist(),
        'fingerprints':hashes,'complete':True,'query_labels_used':False,'goal_achieved':False,
        'scope':'all15_fixed_source_atomic_training_path_not_prediction'}
    out.write_text(json.dumps(saved,separators=(',',':')),encoding='utf-8')
    print(name,'initial',len(metric['features']),'final',len(fs),'checks',len(fit['checks']),fit['status'],flush=True);return saved


def main():
    threadpool_limits(1);FOLDER.mkdir(exist_ok=True);source=verify(SOURCE);verify(ROOT/'results/atomic-postmetric-deletion')
    names=sorted(source['artifact_hashes']);assert len(names)==15
    reports=[run(n,source) for n in names]
    rows=[{'dataset':r['dataset'],'d':r['d'],'initial_count':len(r['fit']['initial_features']),'final_count':len(r['features']),
        'initial_training_BA':r['fit']['initial_hard_LOO_BA'],'final_training_BA':r['fit']['final_hard_LOO_BA'],
        'checks':len(r['fit']['checks']),'status':r['fit']['status']} for r in reports]
    summary={'rows':rows,'datasets':15,'mean_removed':float(np.mean([1-r['final_count']/r['d'] for r in rows])),
        'dataset_balanced_training_gain_pp':float(np.mean([100*(r['final_training_BA']-r['initial_training_BA']) for r in rows])),
        'goal_achieved':False,'independent_audit_required':True,'scope':'all15_training_path_not_predictive_superiority'}
    (FOLDER/'complete.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(summary,flush=True)


if __name__=='__main__':main()
