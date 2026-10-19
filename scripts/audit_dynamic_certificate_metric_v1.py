"""Independent scalar-coordinate objective, certificates and LP KKT replay."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from scipy.sparse import coo_matrix
from independent_crossgroup_soft5 import independent_groups
from independent_canonical_majority import friend_enemy_bounds
from independent_frs_pair_transition_cell import reconstruct_cell, scalar_cell_evaluation, independent_append_cell_lp
from audit_forest_compact_transfer import equivalent


def independent_soft_objective(A,y,w,eps):
    group=independent_groups(y);classes=np.unique(y);valid=[];total=0.;gradient=np.zeros(len(w))
    lo,hi=np.maximum(0,A-eps),np.minimum(1,A+eps)
    for i in range(len(A)):
        ids=np.flatnonzero(group!=group[i]);same=y[ids]==y[i]
        if len(ids)<5 or not same.any():continue
        valid.append((i,ids,same))
    labels=np.array([y[i] for i,_,_ in valid]);classes=np.unique(labels)
    for i,ids,same in valid:
        C=np.zeros((len(ids),len(w)))
        for a in range(len(w)):
            lower=np.maximum(0,np.maximum(lo[ids,a]-hi[i,a],lo[i,a]-hi[ids,a]))
            upper=np.maximum(abs(lo[i,a]-hi[ids,a]),abs(hi[i,a]-lo[ids,a]))
            C[:,a]=np.where(same,upper,lower)
        D=C@w
        q=np.ones(5) if len(ids)==5 else expit((brentq(
            lambda t:float(expit((t-D)/.05).sum()-5),float(D.min()-2),float(D.max()+2),
            xtol=1e-13,rtol=1e-14)-D)/.05)
        own=q[same].sum()/5;alpha=1/(len(classes)*sum(labels==y[i]))
        total-=alpha*np.log(own+1e-12)
        s=q*(1-q);center=s@C/max(s.sum(),1e-14)
        derivative=(s[same].sum()*center-s[same]@C[same])/(.05*5)
        gradient-=alpha*derivative/(own+1e-12)
    log=np.log(len(w)*(w+1e-12))
    return float(total+.01*np.sum(w*log)),gradient+.01*(log+w/(w+1e-12))


def independent_protection(A,y):
    w=np.ones(A.shape[1])/np.sqrt(A.shape[1]);protected=[]
    for i in range(len(A)):
        value=friend_enemy_bounds(A,y,w,i)
        raw=-np.inf if value is None else np.inf if len(value)<3 else sorted(value)[2]
        bound=raw-1e-10*(1+abs(raw)) if np.isfinite(raw) else raw
        protected.append(bound>1e-10)
    return np.array(protected)


def independent_witness_matrix(A,y,protected,info,counts):
    assert info['eps']==.01 and info['retention']==.9 and info['protection_reference']=='original_full'
    at=np.asarray(info['refresh_at_weights']);matrices=[];targets=[]
    for r in info['witnesses']:
        i=r['object'];friends=r['friends'];ignored=r['excluded_enemies'];targets.append(i)
        assert protected[i] and len(friends)==len(set(friends))==3 and i not in friends
        assert all(y[f]==y[i] for f in friends)
        enemies=np.array([j for j in range(len(y)) if y[j]!=y[i]],int)
        assert len(ignored)==len(set(ignored))==min(2,len(enemies)) and set(ignored)<=set(enemies)
        # Nearest selection is merely an optimization heuristic. Arbitrary
        # distinct same-label friends with positive all-enemy contrasts prove
        # the identical robust strict-majority statement.
        enemies=np.array([e for e in enemies if e not in ignored],int)
        C=np.zeros((len(enemies),3,A.shape[1]))
        for a in range(A.shape[1]):
            ilo,ihi=max(0,A[i,a]-.01),min(1,A[i,a]+.01)
            elo,ehi=np.maximum(0,A[enemies,a]-.01),np.minimum(1,A[enemies,a]+.01)
            flo,fhi=np.maximum(0,A[friends,a]-.01),np.minimum(1,A[friends,a]+.01)
            variants=[]
            for t in (ilo,ihi):
                near=np.maximum(0,np.maximum(elo-t,t-ehi));far=np.maximum(abs(t-flo),abs(t-fhi))
                variants.append(near[:,None]-far[None,:])
            C[:,:,a]=np.minimum(variants[0],variants[1])
        matrices.append(C.reshape(-1,A.shape[1]));counts['independent_majority_triplets']+=len(enemies)*3
    assert len(targets)==len(set(targets))
    for label in np.unique(y):
        required=int(np.ceil(.9*sum((y==label)&protected)-1e-12))
        assert sum(y[targets]==label)==required==info['required'][str(label)]
    C=np.concatenate(matrices) if matrices else np.empty((0,A.shape[1]))
    values=C@at;assert not len(C) or np.min(values)>2e-10
    floors=np.minimum(1e-8,(values+2e-10)/2)
    np.testing.assert_allclose(floors,info['floors'],atol=1e-17,rtol=1e-9)
    assert len(C)==info['constraint_rows']
    return C,floors


def check_dynamic_fit(A,y,fit,counts,cell_evidence=None):
    fs=fit['features'];reduced=A[:,fs];p,n,mass=len(fs),len(A),np.sqrt(A.shape[1])
    protected=independent_protection(A,y);assert protected.tolist()==fit['protected']
    assert fit['certificate_eps']==.01 and fit['eta']==.1 and fit['retention']==.9
    assert fit['temperature']==.05 and fit['reg']==.01 and fit['optimization_eps'] in (0.,.01)
    assert fit['original_reference_unchanged']
    zeros=fit['fixed_zeros'];current=np.array(fit['seed']);anchor=np.array(fit['anchor'])
    assert np.all(current[zeros]==0.) and current.min()>=0 and abs(current.sum()-mass)<1e-8
    if not fit['available']:
        equivalent(fit['seed'],fit['weights']);counts['unavailable_original_reference']+=1;return
    cell=fit['class_cell'];equivalent(cell['full_weights'],fit['full_reference']);equivalent(cell['anchor'],fit['anchor'])
    if cell_evidence is None:
        SC,SR=reconstruct_cell(A,y,cell,counts)
    else:
        expected,SC,SR=cell_evidence;equivalent(cell,expected)
    obj=lambda w:independent_soft_objective(reduced,y,np.asarray(w)/mass,fit['optimization_eps'])
    equivalent(obj(current)[0],fit['initial_objective'])
    for step in fit['steps']:
        equivalent(current.tolist(),step['input_weights']);value,gradient=obj(current)
        np.testing.assert_allclose(value,step['input_objective'],atol=1e-8,rtol=1e-8)
        np.testing.assert_allclose(gradient,step['gradient'],atol=1e-7,rtol=1e-7)
        PC,PR=independent_witness_matrix(reduced,y,protected,step['witness_cell'],counts)
        C=np.vstack((PC,SC));rhs=np.r_[PR,SR]
        assert not len(C) or np.all(C@current>rhs)
        if not step['lp_success']:
            assert not step['accepted'];counts['direction_LP_failures']+=1;continue
        rows,cols,vals,b=[],[],[],[]
        def append(ids,values,bound):
            rows.extend([len(b)]*len(ids));cols.extend(ids);vals.extend(values);b.append(bound)
        for j in range(p):
            append([j,p+n+j],[1.,-1.],anchor[j]/mass)
            append([j,p+n+j],[-1.,-1.],-anchor[j]/mass)
        for c,bound in zip(C,rhs):append(list(range(p)),(-c*mass).tolist(),-bound)
        size,extras=independent_append_cell_lp(cell,mass,p,n,append)
        matrix=coo_matrix((vals,(rows,cols)),shape=(len(b),size)).tocsr();b=np.array(b)
        cost=np.r_[gradient,np.zeros(size-p)];eq=np.r_[np.ones(p),np.zeros(size-p)]
        lower=np.r_[np.zeros(2*p+n),[-np.inf if lo is None else lo for lo,_ in extras]]
        upper=np.r_[np.ones(p),np.zeros(n),np.full(p,np.inf),[np.inf if hi is None else hi for _,hi in extras]];upper[zeros]=0.
        record=step['lp'];x=np.array(record['x']);lam=np.array(record['inequality_marginals'])
        sigma=record['equality_marginals'][0];low=np.array(record['lower_marginals']);high=np.array(record['upper_marginals'])
        assert np.max(matrix@x-b)<=1e-6 and abs(eq@x-1)<=1e-7
        assert np.all(x[zeros]==0.) and np.all(x>=lower-1e-7) and np.all(x<=upper+1e-7)
        assert lam.max()<=1e-7 and low.min()>=-1e-7 and high.max()<=1e-7
        assert np.all(abs(low[~np.isfinite(lower)])<=1e-9) and np.all(abs(high[~np.isfinite(upper)])<=1e-9)
        np.testing.assert_allclose(cost-matrix.T@lam-eq*sigma-low-high,0,atol=2e-6,rtol=0)
        dual=float(b@lam+sigma+lower[np.isfinite(lower)]@low[np.isfinite(lower)]+upper[np.isfinite(upper)]@high[np.isfinite(upper)])
        np.testing.assert_allclose(cost@x,record['objective'],atol=1e-6,rtol=0)
        np.testing.assert_allclose(dual,record['objective'],atol=2e-6,rtol=0)
        vertex=np.array(step['vertex']);assert np.all(vertex[zeros]==0.)
        assert not len(C) or np.all(C@vertex>=rhs-1e-14)
        assert scalar_cell_evaluation(cell,vertex)['feasible']
        slope=float(gradient@((vertex-current)/mass));equivalent(slope,step['feasible_direction_slope'])
        selected=None
        for power,trial in enumerate(step['trials']):
            fraction=2.**(-power);assert trial['fraction']==fraction
            candidate=(1-fraction)*current+fraction*vertex;newvalue=obj(candidate)[0]
            np.testing.assert_allclose(newvalue,trial['objective'],atol=1e-8,rtol=1e-8)
            # Small floating differences at an exact FRS boundary may change
            # trial feasibility; accepted candidates must be strictly valid.
            if trial['accepted']:
                assert not len(C) or np.all(C@candidate>rhs-1e-14)
                assert scalar_cell_evaluation(cell,candidate)['feasible'] and np.all(candidate[zeros]==0.)
                assert newvalue<value-1e-10 and newvalue<=value+1e-4*fraction*min(slope,0.)+1e-12
                selected=candidate;assert power==len(step['trials'])-1
        assert step['accepted']==(selected is not None)
        if selected is not None:current=selected
        equivalent(current.tolist(),step['output_weights']);equivalent(obj(current)[0],step['output_objective'])
        counts['independent_direction_primal_dual_replays']+=1
    equivalent(current.tolist(),fit['weights']);equivalent(obj(current)[0],fit['final_objective'])
    C,rhs=independent_witness_matrix(reduced,y,protected,fit['final_witness_cell'],counts)
    assert not len(C) or np.all(C@current>=rhs-1e-14)
    assert scalar_cell_evaluation(cell,current)['feasible']
    assert fit['final_objective']<=fit['initial_objective']+1e-10
    counts['independent_dynamic_certificate_fits']+=1
