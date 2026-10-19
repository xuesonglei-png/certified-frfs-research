"""Independent direct-rank objective, dynamic choices and original FRS replay."""
import hashlib
import numpy as np
from scipy.sparse import coo_matrix
from independent_crossgroup_soft5 import independent_groups
from independent_frs_pair_transition_cell import reconstruct_cell,scalar_cell_evaluation,independent_append_cell_lp
from independent_canonical_polytope import independent_polytope
from independent_canonical_majority import friend_enemy_bounds
from audit_flexible_majority_repair_admission import independent_protected
from audit_forest_compact_transfer import equivalent


def rank_report(A,y,fs,weights,margin):
    groups=independent_groups(y);mass=A.shape[1]**.5;rows=[]
    for i in range(len(A)):
        refs=[j for j in range(len(A)) if groups[j]!=groups[i]]
        distance=lambda j:sum(abs(A[j,a]*weights[k]/mass-A[i,a]*weights[k]/mass) for k,a in enumerate(fs))
        friend=sorted((distance(j),j) for j in refs if y[j]==y[i])
        enemy=sorted((distance(j),j) for j in refs if y[j]!=y[i])
        enough=len(refs)>=5 and len(friend)>=3
        gap=None if not enough else np.inf if len(enemy)<3 else enemy[2][0]-friend[2][0]
        rows.append({'query':i,'qualified':bool(enough and gap>=margin),
            'gap':None if gap is None else 'infinite' if np.isinf(gap) else float(gap),
            'three_friends':[j for _,j in friend[:3]],'two_allowed_enemies':[j for _,j in enemy[:2]]})
    alpha=[1/(len(set(y))*sum(y==c)) for c in y]
    return {'rows':rows,'balanced_qualified_rate':sum(a*r['qualified'] for a,r in zip(alpha,rows,strict=True)),
        'groups':groups.tolist(),'alpha':alpha,'margin':margin,'optimization_eps':0.,
        'scope':'finite_crossgroup_nominal_strict_majority_with_margin_not_exact_plurality_error'}


def check_fit(A,y,fit,counts):
    A,y=np.asarray(A,float),np.asarray(y);fs=fit['features'];anchor=np.asarray(fit['anchor']);seed=np.asarray(fit['seed']);point=np.asarray(fit['weights'])
    p,n,mass=len(fs),len(A),A.shape[1]**.5;zeros=fit['fixed_zero_indices'];margin,reg=fit['margin'],fit['reg']
    assert fit['optimization_eps']==0. and fit['certificate_eps']==.01 and fit['class_cell']['eta']==.1 and fit['polytope']['retention']==.9
    assert not fit['optimization_proven_global'] and fit['mass']==mass
    assert zeros==sorted(set(zeros)) and all(int(j)==j and 0<=j<p for j in zeros)
    assert np.all(seed[zeros]==0.) and np.all(point[zeros]==0.) and point.min()>=0 and abs(sum(point)-mass)<1e-8
    original=independent_protected(A,y,np.ones(A.shape[1])/mass);own=np.zeros(n,bool)
    for i in range(n):
        v=friend_enemy_bounds(A[:,fs],y,anchor,i)
        if v is not None:
            raw=np.partition(v,2)[2];own[i]=raw-1e-10*(1+abs(raw))>1e-10
    ok=all(sum((y==c)&original&own)>=int(np.ceil(.9*sum((y==c)&original)-1e-12)) for c in np.unique(y))
    assert fit['polytope']['protection_reference']==('original_full' if ok else 'own_anchor')
    C,rhs=independent_polytope(A[:,fs],y,anchor,original if ok else own,fit['polytope'])
    SC,SR=reconstruct_cell(A,y,fit['class_cell'],counts);C=np.vstack((C,SC));rhs=np.r_[rhs,SR]
    equivalent(fit['class_cell']['anchor'],anchor.tolist())
    equivalent(fit['class_cell']['full_weights'],(np.ones(A.shape[1])/mass).tolist())
    assert not len(C) or np.all(C@seed>rhs)
    assert all(v<.1 for v in scalar_cell_evaluation(fit['class_cell'],seed)['raw_upper_bounds_with_inflation'])
    assert scalar_cell_evaluation(fit['class_cell'],point)['feasible'] and (not len(C) or np.all(C@point>=rhs))
    seed_report=rank_report(A,y,fs,seed,margin);final_report=rank_report(A,y,fs,point,margin)
    equivalent(seed_report,fit['seed_report']);equivalent(final_report,fit['final_report'])
    objective=lambda w,r:1-r['balanced_qualified_rate']+reg*sum(abs(w/mass-anchor/mass))
    equivalent(objective(seed,seed_report),fit['initial_objective']);equivalent(objective(point,final_report),fit['final_objective'])
    if fit['accepted_descent']:
        assert fit['final_objective']<fit['initial_objective']-1e-10
    else:equivalent(seed.tolist(),point.tolist())
    if fit['candidate_weights'] is not None:
        candidate=np.asarray(fit['candidate_weights'])
        assert candidate.min()>=0 and abs(sum(candidate)-mass)<1e-8 and np.all(candidate[zeros]==0.)
        assert scalar_cell_evaluation(fit['class_cell'],candidate)['feasible'] and (not len(C) or np.all(C@candidate>=rhs))
        candidate_report=rank_report(A,y,fs,candidate,margin)
        equivalent(candidate_report,fit['candidate_report'])
        accepted=objective(candidate,candidate_report)<fit['initial_objective']-1e-10
        assert fit['accepted_descent']==accepted
        if accepted:equivalent(candidate.tolist(),point.tolist())
    else:
        assert not fit['accepted_descent'] and fit['candidate_report'] is None
    rows,cols,values,b=[],[],[],[]
    def append(ids,vals,bound):
        rows.extend([len(b)]*len(ids));cols.extend(ids);values.extend(vals);b.append(bound)
    for j in range(p):
        append([j,p+n+j],[1.,-1.],anchor[j]/mass);append([j,p+n+j],[-1.,-1.],-anchor[j]/mass)
    for c,bound in zip(C,rhs):append(list(range(p)),(-c*mass).tolist(),-bound)
    size,extras=independent_append_cell_lp(fit['class_cell'],mass,p,n,append)
    lower=[0.]*p+[0.]*n+[0.]*p+[(-np.inf if lo is None else lo) for lo,_ in extras]
    upper=[0. if j in zeros else 1. for j in range(p)]+[1.]*n+[np.inf]*(p+len(extras))
    integer_ids=list(range(p,p+n));groups=independent_groups(y);M=1+margin
    assert len(fit['declarations'])==n
    for i,r in enumerate(fit['declarations']):
        refs=[j for j in range(n) if groups[j]!=groups[i]];friends=[j for j in refs if y[j]==y[i]];enemies=[j for j in refs if y[j]!=y[i]]
        assert r['query']==i and r['friends']==friends and r['enemies']==enemies
        if len(refs)<5 or len(friends)<3:upper[p+i]=0.
        assert r['threshold_index']==size;threshold=size;size+=1;lower.append(0.);upper.append(1.)
        fids=[];eids=[]
        for friend in friends:
            selected=size;size+=1;lower.append(0.);upper.append(1.);integer_ids.append(selected);fids.append(selected)
            dist=[abs(A[friend,a]-A[i,a]) for a in fs]
            append(list(range(p))+[threshold,selected],dist+[-1.,M],M)
        append([p+i]+fids,[3.]+[-1.]*len(fids),0.)
        for enemy in enemies:
            ignored=size;size+=1;lower.append(0.);upper.append(1.);integer_ids.append(ignored);eids.append(ignored)
            dist=[-abs(A[enemy,a]-A[i,a]) for a in fs]
            append(list(range(p))+[threshold,ignored,p+i],dist+[1.,-M,M],M-margin)
        append(eids,[1.]*len(eids),2.)
        assert r['friend_selection_indices']==fids and r['enemy_exception_indices']==eids
    matrix=coo_matrix((values,(rows,cols)),shape=(len(b),size)).tocsr();b=np.asarray(b)
    assert hashlib.sha256(matrix.data.tobytes()+matrix.indices.tobytes()+matrix.indptr.tobytes()).hexdigest()==fit['matrix_sha256']
    if fit['mip_incumbent'] is not None:
        raw=np.asarray(fit['mip_incumbent']);assert raw.shape==(size,) and np.isfinite(raw).all()
        primal=bool(np.max(matrix@raw-b)<=1e-6 and abs(sum(raw[:p])-1)<=1e-7
            and np.all(raw>=np.asarray(lower)-1e-7) and np.all(raw<=np.asarray(upper)+1e-7)
            and max(abs(raw[integer_ids]-np.rint(raw[integer_ids])))<=1e-6 and np.all(raw[zeros]==0.))
        if fit['accepted_descent']:assert primal
        alpha=np.asarray(seed_report['alpha']);c=np.r_[np.zeros(p),-alpha,np.full(p,reg),np.zeros(size-2*p-n)]
        equivalent(float(c@raw),fit['mip_objective'])
        counts['independent_mip_primal_replays']+=1
    counts['independent_direct_majority_fit_replays']+=1
