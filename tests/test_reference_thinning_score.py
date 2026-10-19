import numpy as np
import pytest
from reference_thinning_score import reference_masks,thinning_score


def test_shared_thinning_exactly_matches_scalar_loo_votes_including_ties():
    A=np.array([[i%4,(i*3)%5] for i in range(13)],float)
    y=np.array([0,1,2,0,1,2,0,1,2,0,1,2,0]); weights=[1.,2.]
    masks=reference_masks(len(A)); result=thinning_score(A,y,[0,1],weights,masks)
    for mask,draw in zip(masks,result['draws'],strict=True):
        predictions=[]
        for i in range(len(A)):
            eligible=[j for j in range(len(A)) if j!=i and mask[j]]
            neighbors=sorted(eligible,key=lambda j:(sum(weights[a]*abs(A[i,a]-A[j,a]) for a in range(2)),j))[:5]
            predictions.append(max(np.unique(y),key=lambda c:sum(y[j]==c for j in neighbors)))
        assert predictions==draw['predictions']
        assert draw['BA']==pytest.approx(np.mean([np.mean(np.array(predictions)[y==c]==c) for c in np.unique(y)]))


def test_full_retention_is_hard5_and_query_is_excluded_even_if_retained():
    A=np.arange(7,dtype=float)[:,None]; y=np.array([0,0,0,1,1,1,1])
    masks=reference_masks(7,retain=1,draws=2)
    result=thinning_score(A,y,[0],[1.],masks)
    assert np.all(masks) and result['draws'][0]==result['draws'][1]
    assert result['draws'][0]['predictions']==[1,1,1,0,1,1,1]


def test_masks_label_free_deterministic_and_invalid_reference_count_rejected():
    assert np.array_equal(reference_masks(19),reference_masks(19))
    with pytest.raises(ValueError): reference_masks(6)
    with pytest.raises(ValueError): thinning_score(np.zeros((7,2)),np.zeros(7),[0],[1],np.ones((1,7),dtype=int))
