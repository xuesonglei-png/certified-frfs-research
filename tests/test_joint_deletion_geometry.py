import numpy as np
from certified_frfs.geometry_only_forest_stage import GeometryStageContext
from certified_frfs.joint_deletion_geometry import JointDeletionGeometryContext
from certified_frfs.feasible_transport import weight_certificate


def test_joint_jump_reaches_certified_support_inaccessible_to_single_deletions():
    # Rational 3-decimal table from the independently verified Q(sqrt(5))
    # counterexample. No random search or result-folder dependency in this test.
    X = np.array([[0,.327,.405,.692,.922],[.337,.781,0,.081,.034],
        [.176,.420,.576,.510,.335],[.099,.620,.789,1,.567],
        [.241,.623,.697,.308,.089],[.292,0,.433,.176,.073],
        [.566,.591,.738,.676,.676],[.577,.128,1,.710,.394],
        [.280,1,.472,.764,1],[1,.779,.102,0,0],
        [.452,.541,.629,.648,.678],[.660,.753,.391,.752,.691]])
    y = np.repeat([0,1],6)
    old = GeometryStageContext(X,y).fit_stage(C=1.,pair_penalty=None)
    assert len(old['path']['trace']) == 1 and old['policies']['0.5']['fit']['features'] == list(range(5))
    context = JointDeletionGeometryContext(X,y); new = context.fit_stage(C=1.,pair_penalty=None)
    assert new['path']['accepted_pair_jumps'] == 1
    assert len(new['policies']['0.5']['fit']['features']) == 3
    assert new['original_majority_protected_counts'] == old['original_majority_protected_counts'] == {'0':4,'1':3}
    assert all(not a['accepted'] for a in new['path']['attempts'][:5])
    for r in new['path']['trace']:
        fs = r['features']; weights = np.array(r['weights'])[fs]
        verified = weight_certificate(context.A,y,fs,context.original_weights,weights)
        assert verified['feasible'] and verified == r['certificate']
