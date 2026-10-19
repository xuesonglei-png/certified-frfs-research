import numpy as np
from certified_frfs.fuzzy_teacher_support import fuzzy_score_supports

def test_fuzzy_support_keeps_decisive_raw_attribute():
    rng=np.random.default_rng(4);X=rng.random((80,3));y=(X[:,0]>.5).astype(int)
    coef=np.zeros((2,3,3));coef[1,0]=[1.,0.,1.]
    rows=fuzzy_score_supports(X,y,coef,[0.,0.],fractions=(.5,))
    assert len(rows)==1 and 0 in rows[0]['features']
