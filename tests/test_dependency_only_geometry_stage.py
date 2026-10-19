import numpy as np
from certified_frfs.dependency_only_geometry_stage import DependencyOnlyGeometryContext
from certified_frfs.feasible_transport import weight_certificate


def test_dependency_only_admission_does_not_relabel_original_certificate():
    X = np.array([[0,.327,.405,.692,.922],[.337,.781,0,.081,.034],
        [.176,.420,.576,.510,.335],[.099,.620,.789,1,.567],
        [.241,.623,.697,.308,.089],[.292,0,.433,.176,.073],
        [.566,.591,.738,.676,.676],[.577,.128,1,.710,.394],
        [.280,1,.472,.764,1],[1,.779,.102,0,0],
        [.452,.541,.629,.648,.678],[.660,.753,.391,.752,.691]])
    y = np.repeat([0,1],6); context = DependencyOnlyGeometryContext(X,y)
    stage = context.fit_policy(C=1.,pair_penalty=None,target_count=4)
    candidate = stage['candidate']; assert len(candidate['features']) == 4
    assert candidate['dependency_only_admissible'] and not candidate['certificate']['feasible']
    assert not candidate['certificate']['majority_feasible'] and not stage['majority_guard_enforced']
    fs = candidate['features']; weights = np.array(candidate['weights'])[fs]
    independent = weight_certificate(context.A,y,fs,context.original_weights,weights)
    assert independent == candidate['certificate'] and independent['dependency_certificate']['upper_bound'] <= .1
