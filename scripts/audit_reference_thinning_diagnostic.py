"""Separate Manhattan backend and scalar all-row/all-draw vote reconstruction."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances
from threadpoolctl import threadpool_limits
from run_reference_thinning_diagnostic import ROOT,FOLDER,RESCALE,ATOMIC,digest,inputs,candidate_points,fingerprints,summarize
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from continue_soft_topk_knn_pilot import verify


def replay(A,y,fs,w,masks,stored):
    D=manhattan_distances(A[:,fs]*np.array(w)); classes=sorted(set(y.tolist()))
    ordered=np.argsort(D,axis=1,kind='stable'); fresh=[]
    for mask,draw in zip(masks,stored['draws'],strict=True):
        predictions=[]
        for i in range(len(A)):
            neighbors=[]
            for j in ordered[i]:
                if i!=j and mask[j]: neighbors.append(int(j))
                if len(neighbors)==5: break
            assert len(neighbors)==5 and i not in neighbors
            counts=[sum(y[j]==c for j in neighbors) for c in classes]
            predictions.append(classes[counts.index(max(counts))])
        assert predictions==draw['predictions']
        ba=float(np.mean([np.mean(np.array(predictions)[y==c]==c) for c in classes]))
        assert abs(ba-draw['BA'])<1e-12; fresh.append(ba)
    assert abs(np.mean(fresh)-stored['mean_BA'])<1e-12
    assert stored['scope']=='training_reference_thinning_not_crossvalidation'


def main():
    threadpool_limits(1); ra,aa=verify(RESCALE),verify(ATOMIC); reports=[]; unique=0
    for name in sorted(ra['artifact_hashes']):
        path=FOLDER/(name+'.json'); saved=json.loads(path.read_text())
        assert saved['complete'] and not saved['query_labels_used'] and saved['fingerprints']==fingerprints(name)
        assert digest(RESCALE/(name+'.json'))==ra['artifact_hashes'][name]
        assert digest(ATOMIC/(name+'.json'))==aa['artifact_hashes'][name]
        old,atomic=inputs(name); assert saved['reference_indices']==old['reference_indices']
        X,y=load_dataset(name); ri=np.array(saved['reference_indices']); A,_=preprocess_fit(X[ri],X[ri][:1]); labels=y[ri]
        masks=np.array(saved['masks'],bool); assert masks.shape==(8,len(A)) and np.all(masks.sum(axis=1)>=6)
        rng=np.random.default_rng(20261101); expected=[]
        for _ in range(8):
            for attempt in range(10000):
                candidate=rng.random(len(A))<.8
                if candidate.sum()>=6: break
            else: raise AssertionError('mask rejection limit')
            expected.append(candidate.tolist())
        assert expected==saved['masks']; cache={}
        def check(fs,w,score):
            key=(tuple(fs),tuple(w))
            if key not in cache:
                replay(A,labels,fs,w,masks,score);cache[key]=score
            assert score==cache[key]
        assert len(saved['records'])==len(old['fit']['checks'])==len(atomic['records'])
        for r,parent,other in zip(saved['records'],old['fit']['checks'],atomic['records'],strict=True):
            assert r['source_candidate_id']==parent['id'] and r['points']==candidate_points(parent,other)
            check(parent['features_before'],parent['weights_before'],r['parent_score'])
            assert len(r['scores'])==len(r['points'])
            for p,s in zip(r['points'],r['scores'],strict=True):check(p['features'],p['weights'],s)
            eligible=[i for i,p in enumerate(r['points']) if p['eligible']]
            nonworse=[i for i in eligible if r['scores'][i]['mean_BA']>=r['parent_score']['mean_BA']-1e-12]
            def key(i,stress):
                p=r['points'][i];hard=-round(p['hard_BA'],12)
                return ((-round(r['scores'][i]['mean_BA'],12),hard) if stress else (hard,))+(p['kind'],p['receiver'] if p['receiver'] is not None else -1)
            assert r['hard_selected_id']==(min(eligible,key=lambda i:key(i,False)) if eligible else None)
            assert r['stress_selected_id']==(min(nonworse,key=lambda i:key(i,True)) if nonworse else None)
        assert saved['unique_metrics']==len(cache);unique+=len(cache);reports.append(saved)
        print(name,'independent thinning votes replayed',len(cache),'metrics',flush=True)
    summary=json.loads((FOLDER/'complete.json').read_text());assert summary==summarize(reports)
    result={'passed':True,'datasets':15,'candidate_count':117,'unique_metrics':unique,'draws_per_metric':8,
        'artifact_hashes':{r['dataset']:digest(FOLDER/(r['dataset']+'.json')) for r in reports},
        'summary_sha256':digest(FOLDER/'complete.json'),'auditor_sha256':digest(Path(__file__)),
        'scope':'separate_compiled_Manhattan_backend_scalar_all_votes_not_exact_arithmetic_or_generalization'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result,flush=True)


if __name__=='__main__':main()
