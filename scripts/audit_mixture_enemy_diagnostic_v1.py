"""Independent coordinate supremum replay for the fixed mixture diagnostic.

Verifies explicit simplex weights and their envelope without reusing the
production affine rows or LP. Optimality of the mixture is not required for
the upper-bound property and is not asserted by this audit.
"""
import json
from pathlib import Path
import numpy as np
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from run_mixture_enemy_diagnostic_v1 import ROOT,OUT,BASE,FREEZE,DATASETS,digest,old_pair_bound


def scalar_supremum(li,ui,lj,uj,lk,uk,wf,wr,lam,shared):
    coordinate=[]
    for a in range(len(li)):
        points=[li[a],ui[a]]
        # Include all piecewise-affine breakpoints, including far-distance
        # midpoints; independent from the production four-point construction.
        for v in [lj[a],uj[a],*lk[:,a],*uk[:,a],*((lk[:,a]+uk[:,a])/2)]:
            if li[a]<=v<=ui[a]:points.append(v)
        maximum=-np.inf
        for t in points:
            distances=[abs(t-lj[a]),abs(t-uj[a]),max(0.,lj[a]-t,t-uj[a])]
            for dj in distances:
                total=-wr[a]*dj
                for k in range(len(lam)):
                    distance=dj if k==shared else max(abs(t-lk[k,a]),abs(t-uk[k,a]))
                    total+=lam[k]*wf[a]*distance
                maximum=max(maximum,total)
        coordinate.append(maximum)
    return np.asarray(coordinate)


def main():
    frozen=json.loads(FREEZE.read_text())
    for rel,sha in frozen['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
    pairs=0;rows=[]
    for name in DATASETS:
        r=json.loads((OUT/(name+'.json')).read_text());assert r['source_sha256']==digest(BASE/(name+'.json'))
        assert not r['heldout_labels_used'] and not r['classification_scored']
        X,y=load_dataset(name);ri=np.asarray(r['reference_indices']);A,_=preprocess_fit(X[ri],X[ri]);labels=y[ri]
        lo,hi=np.maximum(0,A-.01),np.minimum(1,A+.01);wf=np.ones(A.shape[1])/np.sqrt(A.shape[1]);wr=np.asarray(r['weights'])
        gains=[]
        for o in r['objects']:
            i=o['object'];witnesses=np.asarray(o['witnesses']);assert np.all(labels[witnesses]!=labels[i])
            old_values=[];new_values=[]
            for record in o['records']:
                j=record['enemy'];assert labels[i]!=labels[j]
                shared=int(np.flatnonzero(witnesses==j)[0]) if j in witnesses else None
                old=old_pair_bound(lo[i],hi[i],lo[j],hi[j],lo[witnesses],hi[witnesses],wf,wr,shared)
                np.testing.assert_allclose(record['old_raw'],old,atol=1e-10,rtol=1e-9)
                mixture=record['mixture'];new=old
                if mixture is not None:
                    lam=np.asarray(mixture['mixture_weights']);assert lam.min()>=0
                    np.testing.assert_allclose(lam.sum(),1.,atol=1e-12,rtol=0.)
                    coordinate=scalar_supremum(lo[i],hi[i],lo[j],hi[j],lo[witnesses],hi[witnesses],wf,wr,lam,shared)
                    np.testing.assert_allclose(coordinate,mixture['coordinate_upper_bounds'],atol=1e-12,rtol=1e-9)
                    np.testing.assert_allclose(coordinate.sum(),mixture['upper_bound_raw'],atol=1e-12,rtol=1e-9)
                    assert coordinate.sum()<=old+1e-10;new=min(old,coordinate.sum());pairs+=1
                np.testing.assert_allclose(new,record['new_raw'],atol=1e-12,rtol=1e-9)
                lower=np.maximum(0,np.maximum(lo[j]-hi[i],lo[i]-hi[j]))@wr
                old_values.append(max(0.,min(1-lower,old)));new_values.append(max(0.,min(1-lower,new)))
            np.testing.assert_allclose(max(old_values),o['old_object_bound'],atol=1e-12,rtol=1e-9)
            np.testing.assert_allclose(max(new_values),o['new_object_bound'],atol=1e-12,rtol=1e-9)
            gains.append(max(old_values)-max(new_values))
        rows.append({'dataset':name,'mean_tightening':float(np.mean(gains)),'strict_tightening':any(g>1e-10 for g in gains)})
        print(name,'independent coordinate supremum PASS',flush=True)
    summary=json.loads((OUT/'complete.json').read_text());assert pairs==summary['LPs']
    np.testing.assert_allclose(np.mean([r['mean_tightening'] for r in rows]),summary['mean_object_tightening'],atol=1e-12)
    assert sum(r['strict_tightening'] for r in rows)==summary['datasets_with_strict_tightening']
    result={'passed':True,'datasets':len(rows),'mixtures_replayed':pairs,'rows':rows,
        'artifact_hashes':{n:digest(OUT/(n+'.json')) for n in DATASETS},'summary_sha256':digest(OUT/'complete.json'),
        'auditor_sha256':digest(Path(__file__)),'goal_achieved':False,
        'scope':'independent_simplex_coordinate_upper_envelope_replay_not_solver_optimality_or_prediction'}
    (OUT/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('ALL15 mixture replay PASS',pairs,flush=True)


if __name__=='__main__': main()
