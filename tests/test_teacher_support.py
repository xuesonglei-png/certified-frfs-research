import numpy as np
from certified_frfs.teacher_support import reconstruction_supports

def test_reconstruction_discards_irrelevant_coordinates_and_preserves_cap():
    rng=np.random.default_rng(3);X=rng.random((100,4));y=(X[:,0]>.5).astype(int)
    W=np.array([[0.,0.,0.,0.],[3.,0.,0.,0.]])
    rows=reconstruction_supports(X,y,W,[0.,-1.5])
    assert {len(r['features']) for r in rows}=={2,3}
    for r in rows:
        assert 0 in r['features']
        w=np.array(r['weights']);assert abs(w.sum()-2)<1e-10
        assert w.max()<=1+1e-10
        assert np.all(w[list(set(range(4))-set(r['features']))]==0)
