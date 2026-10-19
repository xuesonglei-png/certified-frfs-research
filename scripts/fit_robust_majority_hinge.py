"""Convex cutting-plane target, simplex mass, unchanged deployment guards."""
import hashlib
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from robust_majority_hinge import RobustMajorityHinge
from soft_topk_predictive_metric import anchor_witness_polytope


def fit_majority_hinge_target(A,y,features,anchor,*,constrained=False,margin=.02,reg=.01,max_iter=200,gap_tolerance=1e-7):
    A,y,anchor=np.asarray(A,float),np.asarray(y),np.asarray(anchor,float);fs=list(features)
    if (not fs or fs!=sorted(set(fs)) or any(int(a)!=a or not 0<=a<A.shape[1] for a in fs)
        or anchor.shape!=(len(fs),) or not np.isfinite(reg) or reg<0 or int(max_iter)!=max_iter or max_iter<1
        or not np.isfinite(gap_tolerance) or gap_tolerance<=0):raise ValueError('invalid target support/settings')
    mass=float(np.sqrt(A.shape[1]));initial=anchor/mass
    if abs(initial.sum()-1)>1e-8:raise ValueError('original metric mass required')
    oracle=RobustMajorityHinge(A[:,fs],y,initial,margin=margin);p,n=len(fs),len(A)
    C,rhs,polytope=anchor_witness_polytope(A,y,fs,anchor) if constrained else (np.empty((0,p)),np.empty(0),None)
    cuts=[];keys=set();points=[];trace=[];start=oracle.objective(initial,reg=reg)
    base_c=np.r_[np.zeros(p),oracle.alpha,np.full(p,reg)]
    bounds=[(0,1)]*p+[(r['constant_loss'],r['constant_loss']) if r['constant_loss'] is not None else (0,None) for r in oracle.rows]+[(0,None)]*p
    def add(w,faces):
        pid=len(points);points.append(w.tolist())
        for face in faces:
            if oracle.rows[face['query']]['constant_loss'] is not None:continue
            key=(face['query'],tuple(face['gradient']),face['intercept'])
            if key not in keys:keys.add(key);cuts.append({**face,'point_id':pid})
    def master():
        indices=[];columns=[];values=[];b=[]
        def append(cols,vals,r):
            row=len(b);indices.extend([row]*len(cols));columns.extend(cols);values.extend(vals);b.append(r)
        for j in range(p):
            append([j,p+n+j],[1,-1],initial[j]);append([j,p+n+j],[-1,-1],-initial[j])
        for row,r in zip(C,rhs):append(list(range(p)),(-row*mass).tolist(),-r)
        for cut in cuts:append(list(range(p))+[p+cut['query']],cut['gradient']+[-1],-cut['intercept'])
        matrix=coo_matrix((values,(indices,columns)),shape=(len(b),2*p+n)).tocsr()
        equality=coo_matrix((np.ones(p),(np.zeros(p,int),np.arange(p))),shape=(1,2*p+n)).tocsr()
        fit=linprog(base_c,A_ub=matrix,b_ub=np.array(b),A_eq=equality,b_eq=[1.],bounds=bounds,method='highs')
        return fit,matrix,np.array(b),equality
    current=initial.copy();loss,faces=oracle.oracle(current);add(current,faces);success=False;failure='iteration_budget_exhausted'
    fit=None;upper=start;lower=None
    for iteration in range(max_iter):
        fit,matrix,b,equality=master()
        if not fit.success:failure='master_lp_failed';break
        candidate=np.maximum(fit.x[:p],0);candidate/=candidate.sum()
        slack=C@(candidate*mass)-rhs;anchor_slack=C@anchor-rhs;blend=0.
        if len(C) and np.any(slack<0):
            bad=slack<0;blend=float(min(1.,1.01*np.max(-slack[bad]/(anchor_slack[bad]-slack[bad]))+1e-10))
            candidate=(1-blend)*candidate+blend*initial
        losses,faces=oracle.oracle(candidate);upper=float(oracle.alpha@losses+reg*abs(candidate-initial).sum());lower=float(fit.fun)
        gap=upper-lower;trace.append({'iteration':iteration,'lower':lower,'upper':upper,'gap':gap,'cuts':len(cuts),'numerical_anchor_blend':blend})
        current=candidate
        if gap<=gap_tolerance and (not len(C) or np.all(C@(candidate*mass)>=rhs)):
            success=True;failure='numerical_master_oracle_gap_met';break
        before=len(cuts);add(candidate,faces)
        if len(cuts)==before:failure='duplicate_cut_numerical_stall';break
    info={'features':fs,'mass':mass,'constrained':bool(constrained),'margin':margin,'reg':reg,'max_iter':max_iter,
        'gap_tolerance':gap_tolerance,'polytope':polytope,'training_witnesses':oracle.export(),'cut_points':points,'cuts':cuts,
        'trace':trace,'optimizer_success':success,'optimizer_iterations':len(trace),'status':failure,
        'training_objective_initial':start,'training_objective_final':upper if success else start,
        'numerical_lower_bound':lower,'scope':'convex_training_majority_hinge_target_with_numerical_gap_not_formal_directed_rounding_certificate_or_test_risk'}
    if fit is not None and fit.success:
        info['last_master']={'x':fit.x.tolist(),'objective':float(fit.fun),'inequality_marginals':fit.ineqlin.marginals.tolist(),
            'equality_marginals':fit.eqlin.marginals.tolist(),'lower_marginals':fit.lower.marginals.tolist(),
            'upper_marginals':fit.upper.marginals.tolist(),'inequality_rows':len(b),
            'matrix_sha256':hashlib.sha256(matrix.data.tobytes()+matrix.indices.tobytes()+matrix.indptr.tobytes()).hexdigest()}
    target=current*mass if success else anchor.copy();info['weights']=target.tolist()
    before,_=oracle.oracle(initial);after,_=oracle.oracle(target/mass)
    info['training_error_upper_bound_initial']=float(oracle.alpha@np.minimum(1,before))
    info['training_error_upper_bound_final']=float(oracle.alpha@np.minimum(1,after))
    if success:assert oracle.objective(current,reg=reg)<=start+gap_tolerance
    return target,info
