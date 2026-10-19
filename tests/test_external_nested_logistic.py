import importlib.util
from pathlib import Path
import sys
import numpy as np

scripts=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('external_nested',scripts/'run_external_nested_logistic.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_grouped_outer_and_inner_splits_have_no_patient_overlap():
    X=np.arange(144,dtype=float).reshape(48,3)
    groups=np.repeat(np.arange(24),2);y=np.repeat(np.arange(24)%2,2)
    for tr,te in module.folds(X,y,groups,outer=True):
        assert set(groups[tr]).isdisjoint(groups[te])
        for ti,qi in module.folds(X[tr],y[tr],groups[tr],outer=False):
            assert set(groups[tr][ti]).isdisjoint(groups[tr][qi])
            assert set(tr[ti]).isdisjoint(te) and set(tr[qi]).isdisjoint(te)


def test_singleton_remains_only_in_inner_reference():
    X=np.arange(27,dtype=float).reshape(9,3);y=np.array([0]*4+[1]*4+[2])
    for ti,qi in module.folds(X,y,None,outer=False):
        assert 8 in ti and 8 not in qi
        assert set(ti).isdisjoint(qi)
