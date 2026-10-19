"""Audit the explicit SAT reduction on all 3-variable clause subsets."""
from pathlib import Path
import itertools,json,hashlib,sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.optimize import linprog
from certified_frfs.margin_verification import exact_negative_loo_margin_milp
from threadpoolctl import threadpool_limits

def table(clauses,d=3):
    # Literal sign +1 means x_k; -1 means not x_k.
    friends=[np.r_[np.full(d,.5),0.]]
    for clause in clauses:
        a=np.full(d+1,.5)
        for k,sign in clause:a[k]=0. if sign==1 else 1.
        friends.append(a)
    X=np.vstack([np.r_[np.full(d,.5),0.],friends,np.r_[np.full(d,.5),d/2-.25]])
    eps=np.zeros_like(X);eps[0,:d]=eps[-1,:d]=.5
    y=np.zeros(len(X),int);y[-1]=1
    return X,y,eps

def exact_radius(clauses,d=3):
    # On each Boolean orthant every distance to a 0,.5,1 friend is affine.
    X,_,_=table(clauses,d);friends=X[1:-1]
    best=-np.inf;best_x=None
    for b in itertools.product([0.,1.],repeat=d):
        lo=np.array(b)*.5;hi=lo+.5;mid=(lo+hi)/2
        signs=np.sign(mid[None,:]-friends[:,:d])
        constants=np.sum(-signs*friends[:,:d],axis=1)+friends[:,d]
        Aub=np.c_[-signs,np.ones(len(friends))]
        solved=linprog(np.r_[np.zeros(d),-1.],A_ub=Aub,b_ub=constants,
                       bounds=[*zip(lo,hi),(None,None)],method='highs')
        assert solved.success
        x=solved.x[:d]
        radius=float(np.min(np.abs(x-friends[:,:d]).sum(axis=1)+friends[:,d]))
        if radius>best:best=radius;best_x=x
    return best,best_x

def satisfied(b,clauses):return all(any(b[k]==(sign==1) for k,sign in c) for c in clauses)

def main():
    start=perf_counter();d=3
    allclauses=[list(zip(range(d),signs)) for signs in itertools.product([-1,1],repeat=d)]
    rows=[]
    with threadpool_limits(limits=1):
        for mask in range(1,1<<len(allclauses)):
            clauses=[c for k,c in enumerate(allclauses) if mask&(1<<k)]
            sats=[list(b) for b in itertools.product([0,1],repeat=d) if satisfied(b,clauses)]
            radius,x=exact_radius(clauses,d);margin=d/2-.25-radius
            assert np.isclose(margin,-.25,atol=1e-8) if sats else margin>=.25-1e-8
            rows.append(dict(mask=mask,clauses=clauses,satisfiable=bool(sats),satisfying_assignment=sats[:1],
                             exact_radius=radius,margin=margin,anchor_witness=x.tolist()))
        checks=[]
        for mask in [1,15,127,255]:
            row=next(r for r in rows if r['mask']==mask)
            X,y,eps=table(row['clauses'],d)
            solved=exact_negative_loo_margin_milp(X,y,0,weights=np.ones(d+1),eps=eps,clip=False,time_limit=10)
            assert -solved['upper_bound']<=row['margin']+1e-7<=-solved['lower_bound']+2e-7
            scale=max(1.,d/2-.25)
            normalized=exact_negative_loo_margin_milp(X/scale,y,0,weights=np.ones(d+1)/(d+1),eps=eps/scale,clip=True,time_limit=10)
            expected=row['margin']/(scale*(d+1))
            assert -normalized['upper_bound']<=expected+1e-7<=-normalized['lower_bound']+2e-7
            checks.append(dict(mask=mask,raw_lower=solved['lower_bound'],raw_upper=solved['upper_bound'],
                               normalized_lower=normalized['lower_bound'],normalized_upper=normalized['upper_bound'],
                               expected_margin=expected))
    result=dict(protocol='sat-reduction-exhaustive-three-variable-audit',rows=rows,milp_checks=checks,
                summary=dict(formulas=len(rows),satisfiable=sum(r['satisfiable'] for r in rows),
                    unsatisfiable=sum(not r['satisfiable'] for r in rows),piecewise_lps=8*len(rows),
                    milp_checks=len(checks)*2,seconds=perf_counter()-start),
                caveat='Finite d=3 arithmetic check supports transcription only; general complexity proof is separate',
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target=ROOT/'results/round3/sat-reduction-audit.json'
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['summary'],indent=2))
if __name__=='__main__':main()
