"""Third-cycle margin certificates versus independent per-object MILP."""
from pathlib import Path
import hashlib,json,sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.margin import mixture_enemy_bound,triplet_interval_min
from certified_frfs.margin_verification import exact_negative_loo_margin_milp
from certified_frfs.core import interval_bounds

def encode(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def hashes():return {str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),*sorted((ROOT/'src/certified_frfs').glob('*.py'))]}

def verify(X,y,features,eps,weights=None,clip=True):
    X,y=np.asarray(X,float),np.asarray(y)
    weights=np.ones(X.shape[1])/X.shape[1] if weights is None else np.asarray(weights,float)
    lo,hi=interval_bounds(X,eps,clip)
    rows=[]
    for i in range(len(X)):
        friends=np.flatnonzero((y==y[i])&(np.arange(len(y))!=i))
        enemies=np.flatnonzero(y!=y[i])
        if not len(friends) or not len(enemies):continue
        pairs=[]
        for j in enemies:
            v=mixture_enemy_bound(lo,hi,i,j,friends,features,weights)
            pairs.append(dict(enemy=int(j),**v))
        mixture=min(p['lower_bound'] for p in pairs)
        onehot=min(p['onehot_bound'] for p in pairs)
        solved=exact_negative_loo_margin_milp(X,y,i,features,weights,eps,clip,time_limit=5.)
        exact_lo=-solved['upper_bound']; exact_hi=-solved['lower_bound']
        assert mixture<=exact_hi+2e-7, (mixture,exact_hi)
        assert mixture>=onehot-1e-8
        rows.append(dict(target=i,class_label=y[i].item(),onehot_lower=onehot,mixture_lower=mixture,
                         exact_margin_lower=exact_lo,exact_margin_upper=exact_hi,
                         mixed_pairs=pairs,milp=solved))
    return dict(X=X,y=y,features=features,weights=weights,eps=eps,clip=clip,objects=rows)

def main():
    start=perf_counter();before=hashes()
    cases={}
    with threadpool_limits(limits=1):
        X=np.array([[.5,0],[0,0],[1,0],[.5,.75]])
        eps=np.zeros_like(X);eps[0,0]=eps[3,0]=.5
        cases['strict_mixture_gain']=verify(X,[0,0,0,1],[0,1],eps,[1,1])
        r=cases['strict_mixture_gain']['objects'][0]
        assert np.isclose(r['onehot_lower'],-.25) and np.isclose(r['mixture_lower'],.25)
        X=np.array([[.5,0],[.25,0],[.75,0],[.5,.6]])
        eps=np.zeros_like(X);eps[0,0]=eps[3,0]=.5;eps[1:3,0]=.25
        cases['midpoint_crossing_gap']=verify(X,[0,0,0,1],[0,1],eps,[1,1])
        r=cases['midpoint_crossing_gap']['objects'][0]
        assert np.isclose(r['onehot_lower'],-.4) and np.isclose(r['mixture_lower'],-.15)
        assert np.isclose(r['exact_margin_upper'],.1,atol=1e-7)
        cases['mixture_must_be_shared_across_attributes']=verify([[0,0],[0,1],[1,0],[.4,.4]],
                                                                 [0,0,0,1],[0,1],0.,[1,1])
        old=json.loads((ROOT/'results/real-subtable-verification.json').read_text(encoding='utf-8'))
        rows=[]
        for entry in old['rows']:
            d=len(entry['X'][0])
            for name,features in [('full',list(range(d))),('half',list(range(d//2)))]:
                item=verify(entry['X'],entry['y'],features,.01)
                item.update(dataset=entry['dataset'],configuration=name);rows.append(item)
            print(entry['dataset'],flush=True)
    obs=[o for r in rows for o in r['objects']]
    stats=dict(tables=15,configurations=30,object_milps=len(obs),optimal_milps=sum(o['milp']['optimal'] for o in obs),
        max_gap=max(o['milp']['gap'] for o in obs),
        onehot_certified=sum(o['onehot_lower']>1e-10 for o in obs),
        mixture_certified=sum(o['mixture_lower']>1e-10 for o in obs),
        exact_certified=sum(o['exact_margin_lower']>1e-10 for o in obs),
        mean_onehot_gap=float(np.mean([o['exact_margin_upper']-o['onehot_lower'] for o in obs])),
        mean_mixture_gap=float(np.mean([o['exact_margin_upper']-o['mixture_lower'] for o in obs])))
    after=hashes();assert before==after
    result=dict(protocol='margin-prototypes-2026-09-20',run_status='complete',source_hashes_before=before,
                source_hashes_after=after,cases=cases,rows=rows,summary=stats,seconds=perf_counter()-start,
                input_sha256=sha(ROOT/'results/real-subtable-verification.json'))
    target=ROOT/'results/round3/margin-prototypes.json'
    target.parent.mkdir(exist_ok=True,parents=True)
    target.write_text(json.dumps(result,indent=2,default=encode,allow_nan=False),encoding='utf-8')
    print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
