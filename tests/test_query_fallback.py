import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_query_support_iteration43 import certified_or_full
from certified_frfs.feasible_transport import weight_certificate


def test_rejected_candidate_falls_back_to_exact_original_full_metric():
    rng=np.random.default_rng(1019);A=rng.uniform(0,1,(16,4));y=np.repeat([0,1],8)
    rejected={'features':[0],'weights':[2.,0.,0.,0.],'config':{'fraction':.35},'certificate':{'feasible':False},'fit':{}}
    fitted,flag=certified_or_full(A,y,rejected)
    assert flag and fitted['features']==[0,1,2,3]
    np.testing.assert_array_equal(fitted['weights'],np.ones(4)/2)
    assert fitted['certificate']['feasible'] and fitted['certificate']['dependency_certificate']['upper_bound']==0.
