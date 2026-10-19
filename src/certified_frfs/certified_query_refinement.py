"""Fixed-support query descent with a fresh certificate at each update."""
import numpy as np
from .bound_direct import _project
from .query_probability import QueryProbabilityObjective
from .validated_pool import preprocess_fit
from .feasible_transport import weight_certificate


def refine_certified_query(X,y,initial,*,steps=10,backtracks=12):
    X,y=np.asarray(X,float),np.asarray(y);d=X.shape[1];full=np.ones(d)/np.sqrt(d);mass=full.sum()
    fs=np.asarray(initial['features'],int);wide=np.asarray(initial['weights'],float)
    if wide.shape!=(d,) or not len(fs) or len(set(fs))!=len(fs) or np.any(fs<0) or np.any(fs>=d) or steps<1 or backtracks<1:
        raise ValueError('invalid initial metric or iteration budget')
    if not np.isfinite(wide).all() or np.any(wide<0) or abs(wide.sum()-mass)>1e-8 or np.any(wide>2*mass/d+1e-8):
        raise ValueError('metric violates mass or cap')
    if set(np.flatnonzero(wide>0))!=set(fs):raise ValueError('support mismatch')
    A,_=preprocess_fit(X,X);obj=QueryProbabilityObjective(X,y);w=wide/mass
    def cert(v):
        ids=np.flatnonzero(v>0).tolist()
        return weight_certificate(A,y,ids,full,v[ids]*mass)
    certificate=cert(w)
    if not certificate['feasible']:raise ValueError('initial metric not certified')
    value,gradient=obj(w);trace=[{'iteration':0,'weights':(w*mass).tolist(),'objective':value,'certificate':certificate}];attempts=[]
    for iteration in range(1,steps+1):
        accepted=None
        for exponent in range(backtracks):
            step=.05*2.**(-exponent);candidate=np.zeros(d)
            candidate[fs]=_project(w[fs]-step*gradient[fs],2/d)
            new_value,new_gradient=obj(candidate);record={'iteration':iteration,'exponent':exponent,'objective':new_value}
            if new_value<value-1e-9:
                new_cert=cert(candidate);record['certificate']=new_cert
                if new_cert['feasible']:accepted=(candidate,new_value,new_gradient,new_cert,step)
            attempts.append(record)
            if accepted is not None:break
        if accepted is None:break
        w,value,gradient,certificate,step=accepted
        trace.append({'iteration':iteration,'weights':(w*mass).tolist(),'objective':value,'certificate':certificate,'step':step})
    return {'features':np.flatnonzero(w>0).tolist(),'weights':(w*mass).tolist(),'certificate':certificate,
        'trace':trace,'attempts':attempts,'training_splits':obj.splits,'steps':steps,'backtracks':backtracks,
        'scope':'fixed_support_certified_query_descent_not_global_optimum_or_test_risk'}
