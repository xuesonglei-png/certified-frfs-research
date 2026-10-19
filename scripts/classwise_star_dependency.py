"""Class-conditional shared-table signed-star FRS dependency loss bounds.

For a fixed class, stable enemy edges form disjoint stars: class objects
are leaves, other-class references are centers. One coefficient is shared
by every edge of a coordinate. Exact maxima use endpoints for positive
coefficients, projected endpoint medians for negative coefficients.
"""
import numpy as np
from certified_frfs.core import _inputs,interval_bounds,_distance,_interval_distances,_exact_stability_mask
from certified_frfs.redistributed_loss import redistributed_loss_bound
from certified_frfs.classwise_dependency_certificate import classwise_dependency_certificate


def signed_star_maximum(lo,hi,edges,coefficient):
    lo,hi=np.asarray(lo,float),np.asarray(hi,float)
    if (lo.ndim!=1 or hi.shape!=lo.shape or not np.isfinite(lo).all() or not np.isfinite(hi).all()
        or np.any(lo>hi) or not np.isfinite(coefficient)):raise ValueError('invalid star intervals or coefficient')
    sources=[int(i) for i,j in edges];centers=[int(j) for i,j in edges]
    if (len(sources)!=len(set(sources)) or set(sources)&set(centers)
        or any(int(i)!=i or int(j)!=j for i,j in edges)
        or any(not 0<=q<len(lo) for q in sources+centers)):raise ValueError('edges must form directed disjoint stars')
    witness=lo.copy();stars=[];value=0.
    for center in sorted(set(centers)):
        leaves=[i for i,j in edges if j==center]
        def far(t):return np.maximum(abs(t-lo[leaves]),abs(t-hi[leaves]))
        candidates=(lo[center],hi[center]);upper=max(float(far(t).sum()) for t in candidates)
        endpoints=np.sort(np.r_[lo[leaves],hi[leaves]])
        median=float(endpoints[len(leaves)-1]);minimum_t=float(np.clip(median,lo[center],hi[center]))
        lower=float(np.maximum(0,np.maximum(lo[leaves]-minimum_t,minimum_t-hi[leaves])).sum())
        if coefficient>=0:
            t=max(candidates,key=lambda q:float(far(q).sum()))
            leaf_values=np.where(abs(t-lo[leaves])>=abs(t-hi[leaves]),lo[leaves],hi[leaves]);maximum=coefficient*upper
        else:t=minimum_t;leaf_values=np.clip(t,lo[leaves],hi[leaves]);maximum=coefficient*lower
        witness[center]=t;witness[leaves]=leaf_values;value+=maximum
        stars.append({'center':center,'leaves':leaves,'H':upper,'L':lower,'center_maximizer':float(t),'maximum':maximum})
    fresh=float(sum(coefficient*abs(witness[i]-witness[j]) for i,j in edges))
    np.testing.assert_allclose(value,fresh,atol=1e-10,rtol=1e-9)
    return {'maximum':float(value),'witness':witness.tolist(),'stars':stars,
            'scope':'exact_real_arithmetic_signed_star_formula_ordinary_float_implementation'}


def classwise_star_dependency(A,y,features,w_full,w_reduced,*,eps=.01,eta=.1,base_dependency=None):
    A,y,fs,wf=_inputs(A,y,features,w_full);wr=np.asarray(w_reduced,float)
    if wr.shape!=(len(fs),) or not np.isfinite(wr).all() or wr.min()<0:raise ValueError('invalid reduced metric')
    base=redistributed_loss_bound(A,y,fs,wf,wr,eps=eps) if base_dependency is None else base_dependency
    if base_dependency is not None and 'redistributed_object_bounds' in base:
        assert base['features']==fs.tolist()
        np.testing.assert_array_equal(base['full_weights'],wf)
        np.testing.assert_array_equal(base['reduced_weights'],wr)
    original=classwise_dependency_certificate(y,{'dependency_certificate':base},eta=eta)
    classes,counts=np.unique(y,return_counts=True)
    if 'redistributed_object_bounds' not in base:
        assert len(classes)==1 or (len(fs)==A.shape[1] and np.array_equal(wr,wf))
        return {'class_certificate':original,'original_class_certificate':original,'global_upper_bound':base['upper_bound'],
            'stable_unsaturated_count':0,'class_reports':[],'scope':'exact_zero_identity_or_single_class'}
    wide=np.zeros(A.shape[1]);wide[fs]=wr;lo,hi=interval_bounds(A,eps)
    cross=y[:,None]!=y[None,:];distance=_distance(A,fs,wide)
    nearest=np.argmin(np.where(cross,distance,np.inf),axis=1)
    stable,margins=_exact_stability_mask(lo,hi,nearest,fs,wide,cross)
    _,ub=_interval_distances(lo,hi,fs,wide,need_lower=False)
    chosen=stable&(ub[np.arange(len(A)),nearest]<=1-1e-12)
    full_stable,_=_exact_stability_mask(lo,hi,nearest,np.arange(A.shape[1]),wf,cross)
    _,full_ub=_interval_distances(lo,hi,np.arange(A.shape[1]),wf,need_lower=False)
    full_exact=full_stable&(full_ub[np.arange(len(A)),nearest]<=1-1e-12)
    coefficient=wf-wide;old_values=np.asarray(base['redistributed_object_bounds']);reports=[];bounds=[]
    for ci,c in enumerate(classes):
        own=y==c;selected=np.flatnonzero(own&chosen);edges=[(int(i),int(nearest[i])) for i in selected]
        assert all(y[i]==c and y[j]!=c for i,j in edges)
        coordinates=[signed_star_maximum(lo[:,a],hi[:,a],edges,float(coefficient[a])) for a in range(A.shape[1])]
        signed_sum=float(sum(r['maximum'] for r in coordinates));remaining=float(old_values[own&~chosen].sum())
        raw=(signed_sum+remaining)/sum(own);inflated=float(np.clip(raw+1e-10*(1+abs(raw)),0,1))
        upper=min(original['upper_bounds'][ci],inflated);bounds.append(upper)
        reports.append({'class':c.item(),'count':int(sum(own)),'selected_indices':selected.tolist(),'edges':edges,
            'coordinates':coordinates,'remaining_objectwise_sum':remaining,'raw_signed_class_loss_bound':raw,
            'star_class_upper_bound':inflated,'original_upper_bound':original['upper_bounds'][ci],
            'upper_bound':upper,'exactness_conditions_hold':bool(np.all(chosen[own]&full_exact[own])),
            'scope':'bound_on_class_mean_signed_dependency_decrease_not_mean_of_individual_positive_losses'})
    cls={**original,'upper_bounds':bounds,'max_upper_bound':max(bounds),'feasible_all_classes':all(q<=eta for q in bounds),
         'scope':'same_finite_reference_joint_box_class_mean_dependency_loss_all_class_prior_mixtures'}
    aggregate=float(np.array(counts)@np.array(bounds)/len(A))
    return {'class_certificate':cls,'original_class_certificate':original,'global_upper_bound':min(base['upper_bound'],aggregate),
        'stable_unsaturated_count':int(chosen.sum()),'class_reports':reports,'nearest_enemies':nearest.tolist(),
        'worst_prior_upper_bound':max(bounds),'exact_all_class_cells':all(r['exactness_conditions_hold'] for r in reports),
        'scope':'classwise_shared_table_signed_star_FRS_dependency_not_classifier_or_population_risk'}
