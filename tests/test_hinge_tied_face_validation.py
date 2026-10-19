import json
from pathlib import Path
import numpy as np
import pytest
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from independent_hinge_cut_validation import validate_cut
from independent_majority_hinge import independent_oracle


def test_observed_tied_friend_face_is_valid_and_corruption_is_rejected():
    root=Path(__file__).resolve().parents[1]
    saved=json.loads((root/'results/majority-hinge-inner-pilot/iris.json').read_text())
    X,_=load_dataset('iris');raw=X[saved['reference_indices']];block=saved['inner'][1]
    A,_=preprocess_fit(raw[block['reference_indices']],raw[block['validation_indices']])
    fit=block['stage']['learners'][3];info=fit['fit']['training_witnesses'];cut=fit['fit']['cuts'][84]
    point=np.array(fit['fit']['cut_points'][cut['point_id']]);table=A[:,fit['features']]
    check=validate_cut(table,point,info,cut)
    assert check['valid_lower_face'] and check['canonical_gradient_differs']
    for other in np.random.default_rng(23).dirichlet(np.ones(len(point)),size=12):
        loss,_=independent_oracle(table,other,{**info,'witnesses':[info['witnesses'][cut['query']]]})
        assert np.array(cut['gradient'])@other+cut['intercept']<=loss[0]+1e-10
    wrong={**cut,'gradient':(np.array(cut['gradient'])+1).tolist()}
    with pytest.raises(AssertionError):validate_cut(table,point,info,wrong)
