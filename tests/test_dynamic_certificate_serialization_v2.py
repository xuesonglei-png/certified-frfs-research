import json
import numpy as np
from run_dynamic_certificate_development_v2 import builtin


def test_numpy_scalars_and_real_arm_summary_roundtrip():
    record={'available':np.bool_(True),'steps':[{'frs':np.bool_(True),'linear':True,
        'accepted':np.bool_(False),'iteration':np.int64(1),'weights':np.array([0.,1.])}],
        'class_cell_final':{'feasible':np.bool_(True),'count':np.int64(12)}}
    restored=json.loads(json.dumps(record,default=builtin))
    assert restored['available'] is True and restored['steps'][0]['accepted'] is False
    assert restored['steps'][0]['iteration']==1 and restored['steps'][0]['weights']==[0.,1.]
