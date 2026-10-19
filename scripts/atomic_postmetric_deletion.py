"""Exact donor deletion via a finite pool of distance-profile receivers.

Use training feature-distance profiles to select up to two receivers. Move
the donor's ENTIRE mass to a receiver and omit the donor coordinate exactly.
Fresh ORIGINAL full global/classwise/majority guards and actual parent hard5
LOO nonworsening are required. This is finite endpoint testing, not a proof
that the whole segment is admissible or that a minimum reduct was found.
"""
import numpy as np
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.classwise_dependency_certificate import classwise_dependency_certificate
from hardaware_guarded_soft5_descent import hard5_LOO_BA


class DistanceProfiles:
    def __init__(self,A):
        A=np.asarray(A,float)
        if A.ndim!=2 or len(A)<7 or not np.isfinite(A).all() or A.min()<0 or A.max()>1:
            raise ValueError('finite scaled training table with at least seven rows required')
        self.A=A.copy(); self.A.setflags(write=False)
        i,j=np.triu_indices(len(A),1)
        self.profiles=np.abs(A[i]-A[j]).T

    def receivers(self,features,donor,*,pool=2):
        if int(pool)!=pool or pool<1: raise ValueError('positive receiver pool required')
        keep=[a for a in features if a!=donor]
        scores=[{'feature':a,'mean_pair_distance_distortion':float(np.mean(abs(self.profiles[a]-self.profiles[donor])))} for a in keep]
        return sorted(scores,key=lambda p:(round(p['mean_pair_distance_distortion'],12),p['feature']))[:pool]


def fit_atomic_deletion(A,y,parent_features,parent_weights,donor,parent_ba,*,geometry=None,protected=None):
    A,y=np.asarray(A,float),np.asarray(y); fs=list(parent_features); w=np.asarray(parent_weights,float)
    if (A.ndim!=2 or len(A)<7 or y.shape!=(len(A),) or fs!=sorted(set(fs)) or len(fs)<2
        or any(int(a)!=a or not 0<=a<A.shape[1] for a in fs) or donor not in fs
        or w.shape!=(len(fs),) or not np.isfinite(w).all() or w.min()<0
        or not np.isfinite(parent_ba) or not 0<=parent_ba<=1):
        raise ValueError('invalid parent support, donor or training accuracy')
    mass=np.sqrt(A.shape[1]); full=np.ones(A.shape[1])/mass
    if abs(w.sum()-mass)>1e-8 or w[fs.index(donor)]<=0:
        raise ValueError('original metric mass and strictly positive donor required')
    if geometry is None: geometry=DistanceProfiles(A)
    if not np.array_equal(geometry.A,A): raise ValueError('distance profiles must match this exact training table')
    receivers=geometry.receivers(fs,donor,pool=2); cfs=[a for a in fs if a!=donor]
    donor_weight=float(w[fs.index(donor)]); checks=[]
    for receiver in receivers:
        cw=np.array([w[fs.index(a)] for a in cfs]); cw[cfs.index(receiver['feature'])]+=donor_weight
        cert=weight_certificate(A,y,cfs,full,cw,protected=protected)
        cls=classwise_dependency_certificate(y,cert)
        actual=hard5_LOO_BA(A[:,cfs],y,cw)
        accepted=bool(cert['feasible'] and cls['feasible_all_classes'])
        nonworse=bool(actual>=parent_ba-1e-12)
        checks.append({**receiver,'features':cfs.copy(),'weights':cw.tolist(),'donor_mass':donor_weight,
            'certificate':cert,'class_certificate':cls,'accepted':accepted,
            'nonworsening_parent_BA':nonworse,'eligible':bool(accepted and nonworse),
            'hard_training_LOO_BA':actual})
    eligible=[i for i,p in enumerate(checks) if p['eligible']]
    selected=min(eligible,key=lambda i:(-round(checks[i]['hard_training_LOO_BA'],12),checks[i]['feature'])) if eligible else None
    return {'parent_features':fs,'parent_weights':w.tolist(),'donor':int(donor),
        'parent_hard_training_LOO_BA':float(parent_ba),'receiver_pool':2,'checks':checks,
        'selected_id':selected,'guard_reference':'original_full','mass':float(mass),
        'scope':'fixed_training_atomic_endpoint_deletion_not_segment_certificate_or_test_risk'}
