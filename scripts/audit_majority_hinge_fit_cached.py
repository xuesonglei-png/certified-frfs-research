"""Independent witnesses, scalar faces, and numerical LP primal/dual audit."""
import hashlib
import numpy as np
from scipy.sparse import coo_matrix
from independent_majority_hinge import independent_oracle
from independent_crossgroup_soft5 import independent_groups
from independent_canonical_polytope import independent_polytope
from independent_canonical_majority import friend_enemy_bounds
from audit_flexible_majority_repair_admission import independent_protected
from audit_forest_compact_transfer import equivalent


def check_fit(A,y,record,counts):
    fs=record['features'];anchor=np.array(record['anchor']);target=np.array(record['target']);fit=record['fit']
    mass=A.shape[1]**.5;initial=anchor/mass;p=len(fs);n=len(A);info=fit['training_witnesses']
    assert fit['margin']==info['margin']==.02 and fit['reg']==.01 and fit['max_iter']==200
    assert fit['gap_tolerance']==1e-7 and info['eps']==.01
    group=independent_groups(y);assert group.tolist()==info['group_assignments'];equivalent(fit['weights'],target.tolist())
    assert abs(anchor.sum()-mass)<1e-8 and abs(target.sum()-mass)<1e-8 and min(target)>=0
    classes=set(y.tolist());alpha=np.array([1/(len(classes)*sum(y==c)) for c in y]);equivalent(alpha.tolist(),info['alpha'])
    for i,row in enumerate(info['witnesses']):
        assert row['query']==i;refs=[j for j in range(n) if group[j]!=group[i]]
        friends=[j for j in refs if y[j]==y[i]];enemies=[j for j in refs if y[j]!=y[i]]
        def distance(j):return sum(abs(A[j,a]*w-A[i,a]*w) for a,w in zip(fs,initial))
        friends=sorted(friends,key=lambda j:(distance(j),j))[:3]
        assert row['references']==refs and row['friends']==friends and row['enemies']==enemies
        constant=1. if len(refs)<5 or len(friends)<3 else 0. if len(enemies)<3 else None
        assert row['constant_loss']==constant
    start,_=independent_oracle(A[:,fs],initial,info);equivalent(float(alpha@start),fit['training_objective_initial'])
    final,_=independent_oracle(A[:,fs],target/mass,info)
    equivalent(float(alpha@np.minimum(1,start)),fit['training_error_upper_bound_initial'])
    equivalent(float(alpha@np.minimum(1,final)),fit['training_error_upper_bound_final'])
    value=float(alpha@final+.01*abs(target/mass-initial).sum());equivalent(value,fit['training_objective_final'])
    C=np.empty((0,p));rhs=np.empty(0)
    if fit['constrained']:
        full=np.ones(A.shape[1])/mass;original=independent_protected(A,y,full);own=np.zeros(n,bool)
        for i in range(n):
            values=friend_enemy_bounds(A[:,fs],y,anchor,i)
            if values is not None:
                v=np.partition(values,2)[2];own[i]=v-1e-10*(1+abs(v))>1e-10
        original_ok=all(sum((y==c)&original&own)>=int(np.ceil(.9*sum((y==c)&original)-1e-12)) for c in np.unique(y))
        assert fit['polytope']['protection_reference']==('original_full' if original_ok else 'own_anchor')
        C,rhs=independent_polytope(A[:,fs],y,anchor,original if original_ok else own,fit['polytope'])
    else:assert fit['polytope'] is None
    from cached_independent_hinge_cut_validation import CachedValidator
    validator=CachedValidator(A[:,fs],info)
    for cut in fit['cuts']:
        point=np.array(fit['cut_points'][cut['point_id']]);i=cut['query'];local={**info,'witnesses':[info['witnesses'][i]]}
        check=validator.validate(point,cut)
        counts['nonunique_hinge_face_replays']+=check['canonical_gradient_differs']
        counts['independent_hinge_cut_replays']+=1
    if 'last_master' in fit:
        master=fit['last_master'];rowids=[];columns=[];values=[];b=[]
        def append(cols,vals,bound):
            rowids.extend([len(b)]*len(cols));columns.extend(cols);values.extend(vals);b.append(bound)
        for j in range(p):
            append([j,p+n+j],[1,-1],initial[j]);append([j,p+n+j],[-1,-1],-initial[j])
        for row,bound in zip(C,rhs):append(list(range(p)),(-row*mass).tolist(),-bound)
        used=fit['trace'][-1]['cuts']
        for cut in fit['cuts'][:used]:append(list(range(p))+[p+cut['query']],cut['gradient']+[-1],-cut['intercept'])
        matrix=coo_matrix((values,(rowids,columns)),shape=(len(b),2*p+n)).tocsr();b=np.array(b)
        assert hashlib.sha256(matrix.data.tobytes()+matrix.indices.tobytes()+matrix.indptr.tobytes()).hexdigest()==master['matrix_sha256']
        assert len(b)==master['inequality_rows'];x=np.array(master['x']);lam=np.array(master['inequality_marginals'])
        equality=np.r_[np.ones(p),np.zeros(n+p)];sigma=master['equality_marginals'][0]
        low=np.array(master['lower_marginals']);high=np.array(master['upper_marginals']);c=np.r_[np.zeros(p),alpha,np.full(p,.01)]
        lower=np.r_[np.zeros(p),[r['constant_loss'] if r['constant_loss'] is not None else 0 for r in info['witnesses']],np.zeros(p)]
        upper=np.r_[np.ones(p),[r['constant_loss'] if r['constant_loss'] is not None else np.inf for r in info['witnesses']],np.full(p,np.inf)]
        assert np.max(matrix@x-b)<1e-6 and abs(equality@x-1)<1e-7
        assert np.min(x-lower)>-1e-7 and np.all(x[np.isfinite(upper)]<=upper[np.isfinite(upper)]+1e-7)
        assert lam.max()<=1e-7 and low.min()>=-1e-7 and high.max()<=1e-7
        np.testing.assert_allclose(c-matrix.T@lam-equality*sigma-low-high,0,atol=1e-6,rtol=0)
        np.testing.assert_allclose(lam*(b-matrix@x),0,atol=1e-6,rtol=0)
        dual=float(b@lam+sigma+lower@low+upper[np.isfinite(upper)]@high[np.isfinite(upper)])
        np.testing.assert_allclose(c@x,master['objective'],atol=1e-6,rtol=0)
        np.testing.assert_allclose(dual,master['objective'],atol=1e-6,rtol=0)
        counts['independent_master_primal_dual_replays']+=1
    if fit['optimizer_success']:
        assert fit['trace'][-1]['gap']<=1e-7 and value<=fit['training_objective_initial']+1e-7
        assert not len(C) or np.all(C@target>=rhs)
    else:equivalent(target.tolist(),anchor.tolist())
    counts['target_optimizer_failures']+=not fit['optimizer_success'];counts['independent_hinge_fit_replays']+=1
