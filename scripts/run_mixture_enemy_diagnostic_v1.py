"""Training-only witness-mixture tightening on every original dataset.

Fixed sample: first 12 training objects in stored order, up to 8 full-enemy
witnesses and the first 12 possible reduced enemies in stable distance order.
Unsolved/unsampled enemies retain the original single-witness envelope. Thus
this diagnostic never silently omits a worst-case enemy from an upper bound.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from mixture_enemy_envelope_v1 import solve_mixture_enemy_bound
from run_robustaware_projection_v1 import DATASETS, BASE

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/mixture-enemy-diagnostic-v1'
FREEZE=ROOT/'research/mixture-enemy-diagnostic-v1-source-freeze.json'


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def old_pair_bound(li,ui,lj,uj,lk,uk,wf,wr,shared):
    # Independent original scalar box formula, not mixture affine-row code.
    values=[]
    for k in range(len(lk)):
        total=0.
        for a in range(len(li)):
            if k==shared:
                coef=wf[a]-wr[a]
                near=max(0.,li[a]-uj[a],lj[a]-ui[a])
                far=max(abs(li[a]-uj[a]),abs(ui[a]-lj[a]))
                total+=coef*(far if coef>=0 else near)
            else:
                points=[li[a],ui[a]]+[v for v in (lj[a],uj[a],lk[k,a],uk[k,a]) if li[a]<=v<=ui[a]]
                total+=max(wf[a]*max(abs(t-lk[k,a]),abs(t-uk[k,a]))-wr[a]*max(0.,lj[a]-t,t-uj[a]) for t in points)
        values.append(total)
    return float(min(values))


def run(name):
    saved=json.loads((BASE/(name+'.json')).read_text());X,y=load_dataset(name)
    ri=np.asarray(saved['reference_indices']);A,_=preprocess_fit(X[ri],X[ri]);labels=y[ri]
    metric=saved['outer']['results']['fgs_weighted120_atomic_count']['metric']
    fs=metric['features'];wf=np.ones(A.shape[1])/np.sqrt(A.shape[1]);wr=np.zeros(A.shape[1]);wr[fs]=metric['weights']
    lo,hi=np.maximum(0,A-.01),np.minimum(1,A+.01);distance=cdist(A*wf,A*wf,'cityblock')
    object_rows=[];pairs=0;started=time.perf_counter()
    for i in range(min(12,len(A))):
        enemies=np.flatnonzero(labels!=labels[i]);
        if not len(enemies): continue
        lower=np.maximum(0,np.maximum(lo[enemies]-hi[i],lo[i]-hi[enemies]))@wr
        upper=np.maximum(abs(lo[i]-hi[enemies]),abs(hi[i]-lo[enemies]))@wr
        possible=enemies[lower<=upper.min()+1e-12]
        witnesses=enemies[np.lexsort((enemies,distance[i,enemies]))][:8]
        order=sorted(possible,key=lambda j:(float(abs(A[j]*wr-A[i]*wr).sum()),int(j)))
        sampled=set(order[:12]);records=[];old_values=[];new_values=[]
        for j in possible:
            shared=int(np.flatnonzero(witnesses==j)[0]) if j in witnesses else None
            single=old_pair_bound(lo[i],hi[i],lo[j],hi[j],lo[witnesses],hi[witnesses],wf,wr,shared)
            bound=single;result=None
            if int(j) in sampled:
                result=solve_mixture_enemy_bound(lo[i],hi[i],lo[j],hi[j],lo[witnesses],hi[witnesses],wf,wr,shared_index=shared)
                np.testing.assert_allclose(single,result['best_single_bound'],atol=1e-10,rtol=1e-9)
                bound=min(single,result['upper_bound_raw']);pairs+=1
            lb=float(lower[np.flatnonzero(enemies==j)[0]])
            old_values.append(max(0.,min(1-lb,single)));new_values.append(max(0.,min(1-lb,bound)))
            records.append({'enemy':int(j),'old_raw':single,'new_raw':bound,'mixture':result})
        object_rows.append({'object':i,'possible_enemies':len(possible),'witnesses':witnesses.tolist(),
            'old_object_bound':max(old_values),'new_object_bound':max(new_values),'records':records})
    row={'dataset':name,'objects':object_rows,'LPs':pairs,'seconds':time.perf_counter()-started,
        'reference_indices':ri.tolist(),'features':fs,'weights':wr.tolist(),
        'source_sha256':digest(BASE/(name+'.json')),'freeze_sha256':digest(FREEZE),
        'heldout_labels_used':False,'classification_scored':False,'goal_achieved':False}
    OUT.mkdir(exist_ok=True);(OUT/(name+'.json')).write_text(json.dumps(row,indent=2),encoding='utf-8')
    improvement=np.mean([r['old_object_bound']-r['new_object_bound'] for r in object_rows])
    print(name,'objects',len(object_rows),'LPs',pairs,'mean tightening',improvement,flush=True)
    return row


def main():
    threadpool_limits(1)
    files=[Path(__file__),ROOT/'scripts/mixture_enemy_envelope_v1.py',ROOT/'tests/test_mixture_enemy_envelope_v1.py']
    files+=sorted((ROOT/'src/certified_frfs').glob('*.py'))
    fingerprints={p.relative_to(ROOT).as_posix():digest(p) for p in files}
    if not FREEZE.exists():FREEZE.write_text(json.dumps({'created_unix':time.time(),'fingerprints':fingerprints},indent=2),encoding='utf-8')
    assert json.loads(FREEZE.read_text())['fingerprints']==fingerprints
    rows=[run(name) for name in DATASETS]
    out={'datasets':15,'LPs':sum(r['LPs'] for r in rows),
        'mean_object_tightening':float(np.mean([np.mean([o['old_object_bound']-o['new_object_bound'] for o in r['objects']]) for r in rows])),
        'datasets_with_strict_tightening':sum(any(o['old_object_bound']>o['new_object_bound']+1e-10 for o in r['objects']) for r in rows),
        'classification_scored':False,'goal_achieved':False,
        'scope':'fixed_training_prefix_witness_mixture_loss_bound_not_full_data_certificate_or_prediction'}
    (OUT/'complete.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__': main()
