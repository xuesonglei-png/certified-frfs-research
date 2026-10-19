"""Independent whole-table enemy maxima and mixture envelope replay."""
import json
from pathlib import Path
import numpy as np
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.redistributed_loss import redistributed_loss_bound
from audit_mixture_enemy_diagnostic_v1 import scalar_supremum
from run_mixture_dependency_development_v1 import ROOT,OUT,BASE,FREEZE,DATASETS,digest


def independent_single_bounds(li,ui,lj,uj,lk,uk,wf,wr,possible,witnesses):
    total=np.zeros((len(possible),len(witnesses)))
    for a in range(len(li)):
        jlo,jhi=lj[:,a,None],uj[:,a,None];klo,khi=lk[None,:,a],uk[None,:,a]
        endpoints=[np.full(total.shape,li[a]),np.full(total.shape,ui[a]),
            np.broadcast_to(np.clip(jlo,li[a],ui[a]),total.shape),
            np.broadcast_to(np.clip(jhi,li[a],ui[a]),total.shape),
            np.broadcast_to(np.clip(klo,li[a],ui[a]),total.shape),
            np.broadcast_to(np.clip(khi,li[a],ui[a]),total.shape)]
        alternatives=[]
        for t in endpoints:
            alternatives.append(wf[a]*np.maximum(abs(t-klo),abs(t-khi))-wr[a]*np.maximum(0,np.maximum(jlo-t,t-jhi)))
        terms=np.maximum.reduce(alternatives)
        coef=wf[a]-wr[a]
        if coef>=0:same=coef*np.maximum(abs(li[a]-jhi),abs(ui[a]-jlo))
        else:same=coef*np.maximum(0,np.maximum(li[a]-jhi,jlo-ui[a]))
        total+=np.where(possible[:,None]==witnesses[None,:],same,terms)
    return total.min(axis=1)


def main():
    freeze=json.loads(FREEZE.read_text())
    for rel,sha in freeze['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
    totals=0;objects=0;rows=[]
    for name in DATASETS:
        r=json.loads((OUT/(name+'.json')).read_text()); assert not r['classification_scored'] and not r['heldout_labels_used']
        source=json.loads((BASE/(name+'.json')).read_text());assert digest(BASE/(name+'.json'))==r['source_sha256']
        for rel,sha in source['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
        X,y=load_dataset(name);ri=np.asarray(r['reference_indices']);A,_=preprocess_fit(X[ri],X[ri]);labels=y[ri]
        fs=r['features'];wf=np.ones(A.shape[1])/np.sqrt(A.shape[1]);wr=np.zeros(A.shape[1]);wr[fs]=r['weights']
        old=redistributed_loss_bound(A,labels,fs,wf,r['weights'])
        cert=r['certificate'];np.testing.assert_allclose(old['redistributed_object_bounds'],cert['original_object_bounds'],atol=1e-12)
        lo,hi=np.maximum(0,A-.01),np.minimum(1,A+.01);by_object={}
        for p in cert['pair_records']:by_object.setdefault(p['object'],[]).append(p)
        calculated=[]
        for i in range(len(A)):
            enemies=np.flatnonzero(labels!=labels[i]);lower=np.maximum(0,np.maximum(lo[enemies]-hi[i],lo[i]-hi[enemies]))@wr
            upper=np.maximum(abs(lo[i]-hi[enemies]),abs(hi[i]-lo[enemies]))@wr
            possible=enemies[lower<=upper.min()+1e-12]
            full_distance=sum(abs(A[enemies,a]-A[i,a])*wf[a] for a in range(A.shape[1]))
            witnesses=enemies[np.lexsort((enemies,full_distance))][:8]
            signed=independent_single_bounds(lo[i],hi[i],lo[possible],hi[possible],lo[witnesses],hi[witnesses],wf,wr,possible,witnesses)
            lb=np.array([lower[np.flatnonzero(enemies==j)[0]] for j in possible]);updated=np.maximum(0,np.minimum(1-lb,signed))
            for p in by_object.get(i,[]):
                j=p['enemy'];assert p['witnesses']==witnesses.tolist();index=int(np.flatnonzero(possible==j)[0])
                shared=int(np.flatnonzero(witnesses==j)[0]) if j in witnesses else None
                lam=np.asarray(p['result']['mixture_weights']);assert lam.min()>=0;np.testing.assert_allclose(lam.sum(),1.,atol=1e-12)
                coordinate=scalar_supremum(lo[i],hi[i],lo[j],hi[j],lo[witnesses],hi[witnesses],wf,wr,lam,shared)
                np.testing.assert_allclose(coordinate,p['result']['coordinate_upper_bounds'],atol=1e-12,rtol=1e-9)
                np.testing.assert_allclose(signed[index],p['result']['best_single_bound'],atol=1e-10,rtol=1e-9)
                updated[index]=max(0.,min(1-lb[index],signed[index],coordinate.sum()));totals+=1
            calculated.append(min(updated.max(),old['redistributed_object_bounds'][i]));objects+=1
        np.testing.assert_allclose(calculated,cert['object_bounds'],atol=1e-11,rtol=1e-9)
        raw=float(np.mean(calculated));np.testing.assert_allclose(np.clip(raw+1e-10*(1+abs(raw)),0,1),cert['upper_bound'],atol=1e-12)
        class_upper=[]
        for c in np.unique(labels):
            value=float(np.mean(np.asarray(calculated)[labels==c]));class_upper.append(min(1.,value+1e-10*(1+abs(value))))
        np.testing.assert_allclose(class_upper,cert['class_upper_bounds'],atol=1e-12)
        feasible=cert['upper_bound']<=.1 and max(class_upper)<=.1
        assert r['mixture_joint_feasible']==(r['original_guard']['certificate']['majority_feasible'] and feasible)
        assert cert['LPs']==len(cert['pair_records'])<=2*len(A)
        rows.append({'dataset':name,'old':cert['original_upper_bound'],'new':cert['upper_bound']})
        print(name,'ALL objects independent envelope replay PASS',flush=True)
    summary=json.loads((OUT/'complete.json').read_text());assert totals==summary['LPs']
    np.testing.assert_allclose(np.mean([r['old']-r['new'] for r in rows]),summary['mean_global_tightening'],atol=1e-12)
    report={'passed':True,'datasets':15,'objects':objects,'mixtures_replayed':totals,
        'artifact_hashes':{n:digest(OUT/(n+'.json')) for n in DATASETS},'summary_sha256':digest(OUT/'complete.json'),
        'auditor_sha256':digest(Path(__file__)),'goal_achieved':False,
        'scope':'independent_whole_enemy_maxima_and_simplex_shared_box_replay_original_bound_recomputed'}
    (OUT/'complete-audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
