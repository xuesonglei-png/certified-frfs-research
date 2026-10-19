import numpy as np
from certified_frfs.independent_screen import paired_balanced_loss,select_independent_screen


def test_pairing_and_class_balance():
    y=np.array([0,0,0,1]); reference=y.copy(); candidate=np.array([1,0,0,0])
    result=paired_balanced_loss(reference,candidate,y)
    assert np.isclose(result['mean_loss_delta'],(1/3+1)/2)
    assert result['upper_loss_delta']>=result['mean_loss_delta']


def test_screen_never_enters_discovery_and_fallback_matches_rule():
    X=np.random.default_rng(6).uniform(.1,.9,(24,3)); y=np.repeat([0,1],12)
    result=select_independent_screen(X,y)
    assert not set(result['discovery_indices'])&set(result['screen_indices'])
    assert set(result['discovery_indices'])|set(result['screen_indices'])==set(range(24))
    assert result['accepted']==(result['assessment']['mean_loss_delta']<=.01+1e-12)
    assert result['features']==(result['candidate_features'] if result['accepted'] else [0,1,2])
    assert len(result['discovery_search']['preprocessing'][0]['reference_indices'])<len(result['discovery_indices'])
