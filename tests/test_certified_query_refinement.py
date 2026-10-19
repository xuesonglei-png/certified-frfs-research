import numpy as np
import pytest
from certified_frfs.certified_query_refinement import refine_certified_query


def test_updates_keep_certificate_and_decrease_real_objective():
    rng=np.random.default_rng(126);X=np.r_[rng.uniform(0,.25,(6,4)),rng.uniform(.75,1,(6,4))];y=np.repeat([0,1],6)
    initial={'features':list(range(4)),'weights':[.5]*4}
    fit=refine_certified_query(X,y,initial,steps=3)
    assert all(t['certificate']['feasible'] for t in fit['trace'])
    assert np.all(np.diff([t['objective'] for t in fit['trace']])<0)
    for t in fit['trace']:
        w=np.array(t['weights']);assert abs(w.sum()-2)<1e-10 and w.min()>=0 and w.max()<=1+1e-10


def test_invalid_initial_mass_is_rejected():
    X=np.random.default_rng(1).random((8,4));y=np.repeat([0,1],4)
    with pytest.raises(ValueError):refine_certified_query(X,y,{'features':[0,1,2,3],'weights':[1.]*4})
