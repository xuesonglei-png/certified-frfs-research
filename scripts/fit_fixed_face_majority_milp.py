"""Direct nominal margin-qualified strict-majority training inside original FRS.

Dynamic friend choices avoid the frozen-friend capped-hinge plateau. This is
a finite-table conservative strict-majority objective, not exact multiclass
plurality error, population optimality or guaranteed test improvement.
"""
import hashlib
import time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from independent_crossgroup_soft5 import independent_groups
from soft_topk_predictive_metric import anchor_witness_polytope
from frs_pair_transition_cell import build_class_cell,append_class_cell_lp,evaluate_class_cell
from frs_pair_transition_metric_repair import repair_cell_metric


def strict_majority_report(A,y,fs,point,*,margin=.02):
    A,y,point=np.asarray(A,float),np.asarray(y),np.asarray(point,float)
    mass=np.sqrt(A.shape[1]);groups=independent_groups(y);rows=[]
    for i in range(len(A)):
        refs=np.flatnonzero(groups!=groups[i]);friends=[j for j in refs if y[j]==y[i]];enemies=[j for j in refs if y[j]!=y[i]]
        distance=lambda j:float(abs(A[j,fs]-A[i,fs])@(point/mass))
        friend_dist=sorted((distance(j),int(j)) for j in friends)
        enemy_dist=sorted((distance(j),int(j)) for j in enemies)
        enough=len(refs)>=5 and len(friends)>=3
        gap=None if not enough else (np.inf if len(enemies)<3 else enemy_dist[2][0]-friend_dist[2][0])
        qualified=bool(enough and gap>=margin)
        rows.append({'query':i,'qualified':qualified,'gap':None if gap is None else ('infinite' if not np.isfinite(gap) else float(gap)),
            'three_friends':[j for _,j in friend_dist[:3]],'two_allowed_enemies':[j for _,j in enemy_dist[:2]]})
    alpha=np.asarray([1/(len(np.unique(y))*sum(y==c)) for c in y])
    return {'rows':rows,'balanced_qualified_rate':float(alpha@np.array([r['qualified'] for r in rows])),
        'groups':groups.tolist(),'alpha':alpha.tolist(),'margin':margin,'optimization_eps':0.,
        'scope':'finite_crossgroup_nominal_strict_majority_with_margin_not_exact_plurality_error'}


def fit_majority_face_milp(A,y,features,anchor,*,full_reference,seed=None,fixed_zero_indices=(),
        margin=.02,reg=.01,time_limit=120.):
    A,y,anchor=np.asarray(A,float),np.asarray(y),np.asarray(anchor,float);fs=list(features);zeros=list(fixed_zero_indices)
    assert fs==sorted(set(fs)) and fs and anchor.shape==(len(fs),) and np.isfinite(A).all() and A.min()>=0 and A.max()<=1
    assert zeros==sorted(set(zeros)) and all(int(j)==j and 0<=j<len(fs) for j in zeros)
    assert np.isfinite(margin) and margin>0 and np.isfinite(reg) and reg>=0 and np.isfinite(time_limit) and time_limit>0
    p,n,mass=len(fs),len(A),float(np.sqrt(A.shape[1]));initial=anchor/mass
    seed=anchor.copy() if seed is None else np.asarray(seed,float)
    assert abs(anchor.sum()-mass)<1e-8 and seed.shape==anchor.shape and seed.min()>=0 and abs(seed.sum()-mass)<1e-8 and np.all(seed[zeros]==0.)
    C,rhs,polytope=anchor_witness_polytope(A,y,fs,anchor)
    SC,SR,cell=build_class_cell(A,y,fs,anchor,np.asarray(full_reference,float));C=np.vstack((C,SC));rhs=np.r_[rhs,SR]
    assert cell['anchor_strictly_feasible'] and (not len(C) or np.all(C@seed>rhs))
    assert all(v<.1 for v in evaluate_class_cell(cell,seed)['raw_upper_bounds_with_inflation'])
    rows,cols,values,b=[],[],[],[]
    def append(ids,vals,bound):
        rows.extend([len(b)]*len(ids));cols.extend(ids);values.extend(vals);b.append(bound)
    for j in range(p):
        append([j,p+n+j],[1.,-1.],initial[j]);append([j,p+n+j],[-1.,-1.],-initial[j])
    for c,bound in zip(C,rhs):append(list(range(p)),(-c*mass).tolist(),-bound)
    size,extras=append_class_cell_lp(cell,mass,p,n,append)
    lower=[0.]*p+[0.]*n+[0.]*p+[(-np.inf if lo is None else lo) for lo,_ in extras]
    upper=[0. if j in zeros else 1. for j in range(p)]+[1.]*n+[np.inf]*p+[np.inf]*len(extras)
    integrality=[0]*p+[1]*n+[0]*(p+len(extras));groups=independent_groups(y)
    declarations=[];M=1+margin
    for i in range(n):
        refs=np.flatnonzero(groups!=groups[i]);friends=[int(j) for j in refs if y[j]==y[i]];enemies=[int(j) for j in refs if y[j]!=y[i]]
        if len(refs)<5 or len(friends)<3:upper[p+i]=0.
        threshold=size;size+=1;lower.append(0.);upper.append(1.);integrality.append(0)
        fids=[];eids=[]
        for friend in friends:
            selected=size;size+=1;lower.append(0.);upper.append(1.);integrality.append(1);fids.append(selected)
            dist=abs(A[friend,fs]-A[i,fs])
            append(list(range(p))+[threshold,selected],dist.tolist()+[-1.,M],M)
        append([p+i]+fids,[3.]+[-1.]*len(fids),0.)
        for enemy in enemies:
            ignored=size;size+=1;lower.append(0.);upper.append(1.);integrality.append(1);eids.append(ignored)
            dist=abs(A[enemy,fs]-A[i,fs])
            append(list(range(p))+[threshold,ignored,p+i],(-dist).tolist()+[1.,-M,M],M-margin)
        append(eids,[1.]*len(eids),2.)
        declarations.append({'query':i,'threshold_index':threshold,'friends':friends,'enemies':enemies,
            'friend_selection_indices':fids,'enemy_exception_indices':eids})
    matrix=coo_matrix((values,(rows,cols)),shape=(len(b),size)).tocsr();b=np.asarray(b)
    eq=coo_matrix((np.ones(p),(np.zeros(p,int),np.arange(p))),shape=(1,size)).tocsr()
    alpha=np.array([1/(len(np.unique(y))*sum(y==c)) for c in y])
    objective=np.r_[np.zeros(p),-alpha,np.full(p,reg),np.zeros(size-2*p-n)]
    timer=time.perf_counter()
    solution=milp(objective,integrality=np.asarray(integrality),bounds=Bounds(lower,upper),
        constraints=(LinearConstraint(matrix,-np.inf,b),LinearConstraint(eq,1.,1.)),
        options={'time_limit':float(time_limit),'mip_rel_gap':0.01,'disp':False})
    seed_report=strict_majority_report(A,y,fs,seed,margin=margin)
    actual_objective=lambda point,report:1-report['balanced_qualified_rate']+reg*float(abs(point/mass-initial).sum())
    start=actual_objective(seed,seed_report);point=seed.copy();report=seed_report
    accepted=False;raw=None;trace=[];blend=None;candidate_report=None;candidate_weights=None;reason='no_mip_incumbent_seed_retained'
    if solution.x is not None:
        raw=np.asarray(solution.x)
        # Validate integer and original primal constraints before deployment.
        ints=np.asarray(integrality,dtype=bool)
        primal_ok=bool(np.isfinite(raw).all() and np.max(matrix@raw-b)<=1e-6 and abs(eq@raw-1.)[0]<=1e-7
            and np.all(raw>=np.asarray(lower)-1e-7) and np.all(raw<=np.asarray(upper)+1e-7)
            and np.max(abs(raw[ints]-np.rint(raw[ints])))<=1e-6 and np.all(raw[zeros]==0.))
        reason='mip_primal_rejected_seed_retained'
        if primal_ok:
            candidate=np.maximum(raw[:p],0.)
            candidate/=candidate.sum();candidate,trace,blend=repair_cell_metric(C,rhs,cell,candidate*mass,seed)
            assert np.all(candidate[zeros]==0.)
            candidate_report=strict_majority_report(A,y,fs,candidate,margin=margin)
            candidate_weights=candidate.tolist()
            if actual_objective(candidate,candidate_report)<start-1e-10:
                point,report,accepted=candidate,candidate_report,True;reason='actual_strict_majority_objective_descent'
            else:reason='no_actual_strict_majority_descent_seed_retained'
    assert evaluate_class_cell(cell,point)['feasible'] and (not len(C) or np.all(C@point>=rhs)) and np.all(point[zeros]==0.)
    info={'features':fs,'anchor':anchor.tolist(),'seed':seed.tolist(),'weights':point.tolist(),'fixed_zero_indices':zeros,
        'mass':mass,'margin':margin,'reg':reg,'time_limit':time_limit,'optimization_eps':0.,'certificate_eps':.01,
        'class_cell':cell,'polytope':polytope,'seed_report':seed_report,'final_report':report,'candidate_report':candidate_report,'candidate_weights':candidate_weights,
        'initial_objective':start,'final_objective':actual_objective(point,report),'accepted_descent':accepted,'status':reason,
        'mip_status':int(solution.status),'mip_message':solution.message,'mip_incumbent':None if raw is None else raw.tolist(),
        'mip_objective':None if raw is None else float(solution.fun),'mip_gap_solver_reported':getattr(solution,'mip_gap',None),
        'mip_dual_bound_solver_reported':getattr(solution,'mip_dual_bound',None),
        'declarations':declarations,'matrix_sha256':hashlib.sha256(matrix.data.tobytes()+matrix.indices.tobytes()+matrix.indptr.tobytes()).hexdigest(),
        'primal_feasibility_required':1e-6,'integer_tolerance':1e-6,'repair_trace':trace,'face_seed_blend':blend,
        'fit_seconds':time.perf_counter()-timer,'optimization_proven_global':False,
        'scope':'direct_finite_nominal_strict_majority_training_milp_in_original_certified_face_no_prediction_guarantee'}
    return point,info
