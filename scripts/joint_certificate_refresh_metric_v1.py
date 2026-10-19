"""Prototype joint majority/FRS cell refresh under immutable full reference.

Original full-table protection, radius and FRS reference NEVER change. Refreshing
three friends/two exceptions/protected objects changes the sufficient polytope,
not the required per-class proportion. This is local optimization, not a global
union-of-polyhedra solver or a guarantee of held-out accuracy.
"""
import hashlib
import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from certified_frfs.margin import triplet_interval_min
from crossgroup_soft5_terms import crossgroup_terms
from soft_topk_metric_objective import soft_topk_objective
from frs_pair_transition_cell import build_class_cell, append_class_cell_lp, evaluate_class_cell
from frs_pair_transition_metric_repair import repair_cell_metric
from independent_frs_pair_transition_cell import scalar_cell_evaluation


def triplets(A, i, enemies, friends, eps=.01):
    lo, hi = np.maximum(0., A-eps), np.minimum(1., A+eps)
    return triplet_interval_min(lo[i][None,None,:], hi[i][None,None,:],
        lo[enemies][:,None,:], hi[enemies][:,None,:],
        lo[friends][None,:,:], hi[friends][None,:,:])


def ranked_witnesses(A, y, weights):
    """Small-memory version of the original exact stored-coordinate convention."""
    A, y, weights = np.asarray(A,float), np.asarray(y), np.asarray(weights,float)
    weighted=A*weights; result=[]
    for i in range(len(A)):
        ids=np.flatnonzero((y==y[i]) & (np.arange(len(y))!=i))
        if len(ids)<3:
            result.append(None);continue
        distance=np.sum(abs(weighted[ids]-weighted[i]),axis=1)
        friends=ids[np.lexsort((ids,distance))][:3]
        enemies=np.flatnonzero(y!=y[i])
        C=triplets(A,i,enemies,friends)
        contrast=np.einsum('efp,p->ef',C,weights).min(axis=1)
        order=np.lexsort((enemies,contrast))
        raw=np.inf if len(order)<3 else float(contrast[order[2]])
        bound=raw-1e-10*(1+abs(raw)) if np.isfinite(raw) else raw
        result.append({'object':i,'friends':friends.tolist(),
            'excluded_enemies':enemies[order[:2]].tolist(), 'raw_bound':raw, 'bound':bound})
    return result


def original_protection(A,y):
    weights=np.ones(A.shape[1])/np.sqrt(A.shape[1])
    rows=ranked_witnesses(A,y,weights)
    return np.array([r is not None and r['bound']>1e-10 for r in rows])


def refreshed_cell(A,y,weights,protected,retention=.9):
    """Return None when this nearest-friend construction cannot certify a seed.

    No substitution of an own-anchor reference is permitted. A prior known-valid
    witness cell may still be used by the optimizer if refreshing is unavailable.
    """
    rows=ranked_witnesses(A,y,weights);chosen=[];required={}
    for label in np.unique(y):
        ids=np.flatnonzero((y==label)&protected)
        count=int(np.ceil(retention*len(ids)-1e-12));required[str(label)]=count
        eligible=[int(i) for i in ids if rows[i] is not None and rows[i]['bound']>1e-10]
        if len(eligible)<count:return None
        chosen.extend(sorted(eligible,key=lambda i:(-rows[i]['bound'],i))[:count])
    matrices=[];witnesses=[]
    for i in chosen:
        record=rows[i];enemies=np.flatnonzero(y!=y[i])
        enemies=np.array([e for e in enemies if e not in record['excluded_enemies']],int)
        C=triplets(A,i,enemies,record['friends']).reshape(-1,A.shape[1])
        matrices.append(C);witnesses.append(record)
    C=np.concatenate(matrices) if matrices else np.empty((0,A.shape[1]))
    values=C@weights
    if len(values) and np.min(values)<=2e-10:return None
    floors=np.minimum(1e-8,(values+2e-10)/2)
    info={'witnesses':witnesses,'required':required,'retention':retention,'eps':.01,
        'protection_reference':'original_full', 'refresh_at_weights':weights.tolist(),
        'matrix_sha256':hashlib.sha256(C.tobytes()).hexdigest(),
        'floors':floors.tolist(), 'constraint_rows':len(C)}
    return C,floors,info


def fit_dynamic_certificate(A,y,features,anchor,full_reference,seed,*,fixed_zeros=(),
        refresh=True,refresh_frs=True,optimization_eps=0.,max_steps=12,lp_time_limit=20.):
    A,y,anchor,seed=np.asarray(A,float),np.asarray(y),np.asarray(anchor,float),np.asarray(seed,float)
    fs=list(features);p,n,mass=len(fs),len(A),float(np.sqrt(A.shape[1]));zeros=list(fixed_zeros)
    if (fs!=sorted(set(fs)) or not fs or anchor.shape!=(p,) or seed.shape!=(p,)
        or not np.isfinite(A).all() or A.min()<0 or A.max()>1 or y.shape!=(n,)
        or zeros!=sorted(set(zeros)) or any(j<0 or j>=p for j in zeros)
        or not np.isfinite(seed).all() or seed.min()<0 or abs(seed.sum()-mass)>1e-8
        or abs(anchor.sum()-mass)>1e-8 or not np.all(seed[zeros]==0.)
        or optimization_eps not in (0.,.01) or int(max_steps)!=max_steps or max_steps<1
        or not np.isfinite(lp_time_limit) or lp_time_limit<=0):
        raise ValueError('invalid unchanged-reference conditional-gradient input')
    protected=original_protection(A,y)
    reduced=A[:,fs];poly=refreshed_cell(reduced,y,seed,protected)
    SC,SR,cell=build_class_cell(A,y,fs,anchor,full_reference)
    start=time.perf_counter();info={'features':fs,'anchor':anchor.tolist(),'seed':seed.tolist(),
        'full_reference':np.asarray(full_reference).tolist(),'fixed_zeros':zeros,'mass':mass,
        'refresh':refresh,'refresh_frs':refresh_frs,'optimization_eps':optimization_eps,'certificate_eps':.01,'eta':.1,
        'retention':.9,'temperature':.05,'reg':.01,'max_steps':max_steps,'lp_time_limit':lp_time_limit,
        'protected':protected.tolist(),'class_cell':cell,'steps':[], 'available':poly is not None,
        'original_reference_unchanged':True,'goal_achieved':False}
    if poly is None:
        return seed.copy(),{**info,'status':'original_full_majority_reference_unavailable',
            'weights':seed.tolist(),'fit_seconds':time.perf_counter()-start}
    if not evaluate_class_cell(cell,seed)['feasible'] or (len(SC) and not np.all(SC@seed>SR)):
        raise ValueError('original FRS seed must remain strictly feasible')
    if not scalar_cell_evaluation(cell,seed)['feasible']:
        return seed.copy(), {**info,'available':False,'status':'source_seed_scalar_guard_unavailable',
            'weights':seed.tolist(),'fit_seconds':time.perf_counter()-start}
    terms,group_info=crossgroup_terms(reduced,y,eps=optimization_eps)
    objective=lambda w:soft_topk_objective(w,terms,temperature=.05,reg=.01)
    current=seed.copy();value=objective(current/mass)[0];info['initial_objective']=value
    initial_cell=cell;cell_history=[{'iteration':-1,'available':True,'class_cell':cell}]
    cell_id=0
    info['group_info']=group_info;info['status']='step_budget_exhausted_not_stationarity'
    for iteration in range(max_steps):
        frs_refreshed=False;frs_refresh_reason='disabled_or_initial'
        if refresh_frs and iteration:
            newSC,newSR,newcell=build_class_cell(A,y,fs,current,full_reference)
            newreport=evaluate_class_cell(newcell,current)
            available=bool(newcell['anchor_strictly_feasible']
                and scalar_cell_evaluation(newcell,current)['feasible']
                and newreport['feasible'] and all(v<.1 for v in newreport['raw_upper_bounds_with_inflation'])
                and (not len(newSC) or np.all(newSC@current>newSR)))
            cell_history.append({'iteration':iteration,'available':available,'class_cell':newcell})
            if available:
                SC,SR,cell=newSC,newSR,newcell;cell_id=len(cell_history)-1
                frs_refreshed=True;frs_refresh_reason='fresh_original_reference_certificate'
            else:
                frs_refresh_reason='fresh_bound_unavailable_prior_cell_kept'
        changed=False
        if refresh and iteration:
            new=refreshed_cell(reduced,y,current,protected)
            if new is not None:
                changed=[(r['object'],r['friends'],r['excluded_enemies']) for r in new[2]['witnesses']]!=[(r['object'],r['friends'],r['excluded_enemies']) for r in poly[2]['witnesses']];poly=new
        PC,PR,witness_info=poly;C=np.vstack((PC,SC));rhs=np.r_[PR,SR]
        rows,cols,values,b=[],[],[],[]
        def append(ids,vals,bound):
            rows.extend([len(b)]*len(ids));cols.extend(ids);values.extend(vals);b.append(bound)
        for j in range(p):
            append([j,p+n+j],[1.,-1.],cell['anchor'][j]/mass)
            append([j,p+n+j],[-1.,-1.],-cell['anchor'][j]/mass)
        for c,bound in zip(C,rhs):append(list(range(p)),(-c*mass).tolist(),-bound)
        size,extra=append_class_cell_lp(cell,mass,p,n,append)
        matrix=coo_matrix((values,(rows,cols)),shape=(len(b),size)).tocsr();b=np.array(b)
        bounds=[(0.,0. if j in zeros else 1.) for j in range(p)]+[(0.,0.)]*n+[(0.,None)]*p+extra
        value,gradient=objective(current/mass)
        cost=np.r_[gradient,np.zeros(size-p)]
        eq=coo_matrix((np.ones(p),(np.zeros(p,int),np.arange(p))),shape=(1,size)).tocsr()
        lp=linprog(cost,A_ub=matrix,b_ub=b,A_eq=eq,b_eq=[1.],bounds=bounds,method='highs',
            options={'time_limit':lp_time_limit,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        step={'iteration':iteration,'input_weights':current.tolist(),'input_objective':value,
            'gradient':gradient.tolist(),'witness_cell':witness_info,'witness_refresh_changed':changed,
            'frs_cell_id':cell_id,'frs_refreshed':frs_refreshed,'frs_refresh_reason':frs_refresh_reason,
            'lp_success':bool(lp.success),'lp_status':int(lp.status),'lp_message':lp.message,
            'accepted':False}
        if not lp.success:
            info['status']='direction_LP_failed_current_retained';info['steps'].append(step);break
        v=np.maximum(lp.x[:p],0.);v/=v.sum()
        vertex,trace,blend=repair_cell_metric(C,rhs,cell,v*mass,current)
        scalar_repairs=[]; prior=vertex.copy()
        for attempt in range(12):
            if scalar_cell_evaluation(cell,vertex)['feasible'] and evaluate_class_cell(cell,vertex)['feasible'] and (not len(C) or np.all(C@vertex>rhs)):
                break
            fraction=min(1.,10.**(attempt-10))
            vertex=(1-fraction)*prior+fraction*current
            scalar_repairs.append({'seed_fraction':fraction})
        else:
            vertex=current.copy();scalar_repairs.append({'explicit_current':True})
        assert scalar_cell_evaluation(cell,vertex)['feasible'] and evaluate_class_cell(cell,vertex)['feasible']
        step['scalar_guard_repairs']=scalar_repairs
        assert np.all(vertex[zeros]==0.)
        step['vertex']=vertex.tolist();step['repair_trace']=trace;step['repair_fraction']=blend
        # Save solver evidence; audit reconstructs matrix independently.
        step['lp']={'x':lp.x.tolist(),'objective':float(lp.fun),
            'inequality_marginals':lp.ineqlin.marginals.tolist(),
            'equality_marginals':lp.eqlin.marginals.tolist(),
            'lower_marginals':lp.lower.marginals.tolist(),'upper_marginals':lp.upper.marginals.tolist()}
        slope=float(gradient@((vertex-current)/mass));step['feasible_direction_slope']=slope
        trials=[];accepted=False
        for power in range(21):
            fraction=2.**(-power);candidate=(1-fraction)*current+fraction*vertex
            candidate_value=objective(candidate/mass)[0]
            linear=not len(C) or bool(np.all(C@candidate>rhs))
            frs_report=evaluate_class_cell(cell,candidate)
            scalar_frs=scalar_cell_evaluation(cell,candidate)['feasible']
            frs=frs_report['feasible'] and scalar_frs and all(v<.1 for v in frs_report['raw_upper_bounds_with_inflation'])
            good=bool(linear and frs and np.all(candidate[zeros]==0.)
                and candidate_value<value-1e-10 and candidate_value<=value+1e-4*fraction*min(slope,0.))
            trials.append({'fraction':fraction,'objective':candidate_value,'linear':linear,'frs':frs,'accepted':good})
            if good:
                accepted=True;break
        step['trials']=trials;step['accepted']=accepted
        if accepted:
            current=candidate;value=candidate_value
        step['output_weights']=current.tolist();step['output_objective']=value
        info['steps'].append(step)
        if not accepted:
            info['status']='no_verified_line_search_descent_not_global_stationarity';break
    info.update({'weights':current.tolist(),'final_objective':value,'fit_seconds':time.perf_counter()-start,
        'final_witness_cell':poly[2],'initial_class_cell':initial_cell,
        'class_cell_history':cell_history,'final_class_cell_id':cell_id,'class_cell':cell, 'class_cell_final':evaluate_class_cell(cell,current),
        'accepted_steps':sum(s['accepted'] for s in info['steps']),
        'scope':'local_train_only_refreshed_certificate_union_original_full_reference_not_heldout_guarantee'})
    return current,info
