"""Label-free inner-query soft/hard neighbor mismatch, all15/44 fixed anchors.

No accuracy scoring and no temperature selection. All fixed temperatures are
reported. Independent coordinate distances and scalar Brent thresholds replay
every query; a gap bound is checked, not used as a population risk statement.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from continue_soft_topk_knn_pilot import verify
from run_simplex_soft5_knn_pilot import ROOT,digest
from soft_topk_metric_objective import soft_topk_membership,topk_leakage_bound

SOURCE=ROOT/'results/postmetric-pruning-classifier'
METHODS=('proposed','unguarded_refined','raw_repair_refined','full_refined','fgs_refined','sfrip_refined')
TEMPERATURES=(.05,.02,.01,.005)


def summarize(D,Q,temperature):
    order=np.argsort(D,axis=1,kind='stable');inside=np.take_along_axis(Q,order[:,:5],axis=1).sum(axis=1)
    gap=np.take_along_axis(D,order[:,5:6],axis=1)[:,0]-np.take_along_axis(D,order[:,4:5],axis=1)[:,0]
    leakage=5-inside;bound=np.array([topk_leakage_bound(g,len(D[0]),temperature=temperature) for g in gap])
    assert np.all(leakage<=bound+1e-8)
    effective=25/(Q*Q).sum(axis=1)
    return {'queries':len(D),'references':D.shape[1],'temperature':temperature,
        'mean_hard5_mass_fraction':float(np.mean(inside/5)),
        'mean_outside_hard5_mass_fraction':float(np.mean(leakage/5)),
        'mean_effective_neighbor_count':float(np.mean(effective)),
        'mean_gap_bound_fraction':float(np.mean(bound/5)),
        'median_k_kplus1_gap':float(np.median(gap)),
        'zero_gap_fraction':float(np.mean(gap==0)),
        'queries_below_90pct_hard5_mass':int(np.sum(inside<4.5)),
        'max_mass_residual':float(np.max(abs(Q.sum(axis=1)-5)))}


def replay(A,B,features,weights,D,Q,t):
    independent=np.zeros(D.shape);mass=np.sqrt(A.shape[1])
    for a,w in zip(features,weights):independent+=np.abs(B[:,a,None]*w-A[None,:,a]*w)
    independent/=mass
    np.testing.assert_allclose(independent,D,atol=2e-14,rtol=1e-12)
    fresh=[]
    for row in independent:
        radius=brentq(lambda r:float(expit((r-row)/t).sum()-5),float(row.min()-40*t),
            float(row.max()+40*t),xtol=1e-13,rtol=1e-14)
        q=expit((radius-row)/t);assert abs(q.sum()-5)<1e-8;fresh.append(q)
    fresh=np.array(fresh)
    np.testing.assert_allclose(fresh,Q,atol=2e-10,rtol=1e-9)
    # At exact floating distance ties neighbor identities can differ, but their
    # equal soft mass is interchangeable. Check summaries to numerical tolerance.
    original=summarize(D,Q,t);scalar=summarize(independent,fresh,t)
    for key in ('mean_hard5_mass_fraction','mean_outside_hard5_mass_fraction',
                'mean_effective_neighbor_count','mean_gap_bound_fraction','median_k_kplus1_gap'):
        np.testing.assert_allclose(original[key],scalar[key],atol=1e-9,rtol=1e-9)


def main():
    threadpool_limits(1);audit=verify(SOURCE);rows=[];hashes={}
    for name,sha in sorted(audit['artifact_hashes'].items()):
        path=SOURCE/(name+'.json');assert digest(path)==sha;hashes[name]=sha
        parent=json.loads(path.read_text());X,_=load_dataset(name);raw=X[np.array(parent['reference_indices'])]
        for fi,block in enumerate(parent['inner']):
            ca,cb=np.array(block['reference_indices']),np.array(block['validation_indices'])
            A,B=preprocess_fit(raw[ca],raw[cb]);cache={}
            for method in METHODS:
                metric=block['options'][method][parent['selection']['selected'][method]]['metric']
                fs,w=metric['features'],metric['weights'];key=(tuple(fs),tuple(w))
                if key not in cache:
                    D=cdist(B[:,fs]*w,A[:,fs]*w,'cityblock')/np.sqrt(A.shape[1])
                    reports=[]
                    for t in TEMPERATURES:
                        Q=soft_topk_membership(D,np.ones(D.shape,bool),temperature=t)
                        replay(A,B,fs,w,D,Q,t);reports.append(summarize(D,Q,t))
                    cache[key]=reports
                rows.append({'dataset':name,'inner_fold':fi,'method':method,'reports':cache[key]})
            print(name,'inner',fi+1,'label-free density/scalar replay complete',flush=True)
    means={m:[{'temperature':t,**{key:float(np.mean([np.mean([r['reports'][ti][key] for r in rows if r['dataset']==name and r['method']==m]) for name in hashes]))
        for key in ('mean_hard5_mass_fraction','mean_outside_hard5_mass_fraction','mean_effective_neighbor_count','mean_gap_bound_fraction','median_k_kplus1_gap','zero_gap_fraction')}}
        for ti,t in enumerate(TEMPERATURES)] for m in METHODS}
    result={'passed_independent_scalar_replay':True,'rows':rows,'means':means,
        'source_artifact_hashes':hashes,'source_audit_sha256':digest(SOURCE/'complete-audit.json'),
        'script_sha256':digest(Path(__file__)),'datasets':len(hashes),'inner_folds':len(rows)//len(METHODS),
        'query_labels_used':False,'new_outer_evaluation_performed':False,'temperature_selected':False,'goal_achieved':False,
        'scope':'all15_inner_label_free_neighbor_mass_density_diagnostic_not_accuracy_or_population_risk'}
    assert result['datasets']==15 and result['inner_folds']==44
    (ROOT/'research/soft5-density-mismatch-diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__':main()
