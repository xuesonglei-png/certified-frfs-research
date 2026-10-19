"""Independently reconstruct every ALL15 atomic/rescale training path."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_atomic_pruning_admission import ROOT,FOLDER,SOURCE,fingerprints,digest
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from independent_atomic_pruning import check_atomic_pruning
from audit_forest_compact_transfer import equivalent


def audit(name):
    path=FOLDER/(name+'.json');saved=json.loads(path.read_text());assert saved['complete'] and not saved['query_labels_used']
    assert saved['fingerprints']==fingerprints(name);source=json.loads((SOURCE/(name+'.json')).read_text())
    metric=source['outer']['results']['proposed']['metric'];fit=saved['fit'];assert saved['reference_indices']==source['reference_indices']
    assert fit['initial_features']==metric['features'];equivalent(fit['initial_weights'],metric['weights'])
    assert fit['fraction']==.5 and fit['pool']==4 and fit['max_checks']==48 and fit['guard_reference']=='original_full'
    X,y=load_dataset(name);ri=np.array(saved['reference_indices']);A,_=preprocess_fit(X[ri],X[ri][:1])
    count=check_atomic_pruning(A,y[ri],metric['features'],metric['weights'],fit)
    assert saved['features']==fit['features'];equivalent(saved['weights'],fit['weights'])
    print(name,'complete atomic path independently replayed',count,flush=True);return saved,count


def main():
    threadpool_limits(1);source=verify(SOURCE);names=sorted(source['artifact_hashes']);assert len(names)==15
    reports=[audit(n) for n in names];rows=[{'dataset':r['dataset'],'d':r['d'],'initial_count':len(r['fit']['initial_features']),
        'final_count':len(r['features']),'initial_training_BA':r['fit']['initial_hard_LOO_BA'],
        'final_training_BA':r['fit']['final_hard_LOO_BA'],'checks':c,'status':r['fit']['status']} for r,c in reports]
    summary=FOLDER/'complete.json';stored=json.loads(summary.read_text());equivalent(stored['rows'],rows)
    equivalent(stored['mean_removed'],float(np.mean([1-r['final_count']/r['d'] for r in rows])))
    equivalent(stored['dataset_balanced_training_gain_pp'],float(np.mean([100*(r['final_training_BA']-r['initial_training_BA']) for r in rows])))
    result={'passed':True,'counts':{'datasets':15,'scalar_path_checks':sum(c for _,c in reports)},
        'artifact_hashes':{n:digest(FOLDER/(n+'.json')) for n in names},'summary_sha256':digest(summary),
        'auditor_sha256':digest(Path(__file__)),'goal_achieved':False,
        'limits':'production_shared_dependency_bound_reused_no_query_or_global_optimum_claim'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result,flush=True)


if __name__=='__main__':main()
