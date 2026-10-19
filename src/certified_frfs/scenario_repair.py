"""Fixed-support weight repair against shared measurement scenarios."""
import numpy as np
from scipy.optimize import linprog
from .core import gamma
from .shared_metric_attack import attack_shared_metric
from .feasible_transport import weight_certificate


def solve_scenario_repair(scenarios,y,full,initial,features,*,eta=.09,seconds=5.):
    full,initial=np.asarray(full,float),np.asarray(initial,float);fs=np.asarray(features,int)
    d=len(full);k=len(fs);n=len(y);mass=full.sum();cap=2*mass/d
    if not scenarios or k<1 or len(set(fs))!=k or np.any(fs<0) or np.any(fs>=d) or initial.shape!=(d,) or eta<0:
        raise ValueError('invalid scenario repair inputs')
    # v, absolute changes, then per-scenario clipped nearest-enemy memberships.
    width=2*k+len(scenarios)*n;rows=[];rhs=[]
    for a in range(k):
        row=np.zeros(width);row[a]=1;row[k+a]=-1;rows.append(row);rhs.append(initial[fs[a]])
        row=np.zeros(width);row[a]=-1;row[k+a]=-1;rows.append(row);rhs.append(-initial[fs[a]])
    for s,Z in enumerate(scenarios):
        Z=np.asarray(Z,float)
        if Z.shape!=(n,d) or not np.isfinite(Z).all():raise ValueError('invalid scenario shape')
        for i in range(n):
            for j in np.flatnonzero(np.asarray(y)!=y[i]):
                row=np.zeros(width);row[:k]=-abs(Z[i,fs]-Z[j,fs]);row[2*k+s*n+i]=1
                rows.append(row);rhs.append(0.)
        row=np.zeros(width);row[2*k+s*n:2*k+(s+1)*n]=-1/n
        rows.append(row);rhs.append(eta-gamma(Z,y,weights=full))
    eq=np.zeros((1,width));eq[0,:k]=1
    fit=linprog(np.r_[np.zeros(k),np.ones(k),np.zeros(len(scenarios)*n)],
        A_ub=np.array(rows),b_ub=np.array(rhs),A_eq=eq,b_eq=[mass],
        bounds=[(0,cap)]*k+[(0,mass)]*k+[(0,1)]*(len(scenarios)*n),
        method='highs',options={'time_limit':seconds})
    result={'status':int(fit.status),'success':bool(fit.success),'weights':None,'eta':eta}
    if fit.success:
        v=np.zeros(d);v[fs]=np.maximum(0,fit.x[:k])
        losses=[gamma(Z,y,weights=full)-gamma(Z,y,weights=v) for Z in scenarios]
        valid=abs(v.sum()-mass)<1e-8 and v.max()<=cap+1e-9 and max(losses)<=eta+1e-9
        result.update({'success':bool(valid),'weights':v.tolist() if valid else None,'scenario_losses':losses,
            'shift':float(np.sum(abs(v-initial)))})
    return result


def repair_with_scenarios(X,y,features,full,initial,first_scenario,*,rounds=3):
    X=np.asarray(X,float);full=np.asarray(full,float);v=np.asarray(initial,float)
    low,high=np.maximum(0,X-.01),np.minimum(1,X+.01);first=np.asarray(first_scenario,float)
    if first.shape!=X.shape or np.any(first<low) or np.any(first>high):raise ValueError('scenario outside shared box')
    scenarios=[X.copy(),first.copy()];trace=[]
    for iteration in range(rounds):
        fit=solve_scenario_repair(scenarios,y,full,v,features)
        record={'iteration':iteration,'lp':fit}
        if not fit['success']:
            trace.append(record);break
        v=np.array(fit['weights']);fs=np.flatnonzero(v>0).tolist()
        cert=weight_certificate(X,y,fs,full,v[fs]);record['certificate']=cert
        attack=attack_shared_metric(X,y,full,v);record['attack']=attack;trace.append(record)
        if cert['feasible']:break
        scenarios.append(np.array(attack['perturbed_table']))
    final=trace[-1]
    return {'trace':trace,'weights':v.tolist(),'features':np.flatnonzero(v>0).tolist(),
        'certified':bool(final.get('certificate',{}).get('feasible',False)),
        'scenarios':[Z.tolist() for Z in scenarios],
        'scope':'fixed_support_scenario_LP_fresh_certificate_not_scenario_safety_or_prediction'}
