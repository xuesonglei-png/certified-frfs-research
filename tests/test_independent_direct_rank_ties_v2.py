import copy
import numpy as np
import pytest
from audit_fixed_face_majority_milp import rank_report
from independent_direct_rank_ties_v2 import validate_report


def fixture():
    A=np.array([[x,c] for c in (0.,1.) for x in (0.,0.,.1,.1,.2,.2)])
    return A,np.repeat([0,1],6)


def test_valid_tied_identity_does_not_require_unique_numpy_accumulation_order():
    A,y=fixture();point=np.array([0.,np.sqrt(2)]);report=rank_report(A,y,[0,1],point,.02)
    changed=copy.deepcopy(report);changed['rows'][0]['three_friends'].reverse()
    validate_report(A,y,[0,1],point,.02,changed)
    broken=copy.deepcopy(changed);broken['rows'][0]['qualified']=not broken['rows'][0]['qualified']
    with pytest.raises(AssertionError):validate_report(A,y,[0,1],point,.02,broken)


def test_genuinely_distant_witness_is_rejected_not_called_a_numeric_tie():
    A=np.array([[x,c] for c in (0.,1.) for x in np.linspace(0.,.8,9)]);y=np.repeat([0,1],9)
    point=np.ones(2)/np.sqrt(2);report=rank_report(A,y,[0,1],point,.02);found=False
    for row in report['rows']:
        i=row['query'];ids=row['three_friends'];distance=lambda j:sum(abs(A[j,a]-A[i,a])*point[a]/np.sqrt(2) for a in range(2))
        farther=[j for j in range(len(y)) if report['groups'][j]!=report['groups'][i] and y[j]==y[i] and j not in ids and distance(j)>max(distance(k) for k in ids)+.01]
        if farther:
            broken=copy.deepcopy(report);broken['rows'][i]['three_friends'][-1]=farther[0]
            with pytest.raises(AssertionError):validate_report(A,y,[0,1],point,.02,broken)
            found=True;break
    assert found
