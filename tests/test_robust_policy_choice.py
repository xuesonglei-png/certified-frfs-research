import pytest
from certified_frfs.robust_policy_choice import choose_robust_policy


def row(clean,robust,size,index=0):
    return {'model_index':index,'selector_index':0,'fraction_index':0,
        'folds':[{'clean':clean,'robust':robust,'features':list(range(size))}]}


def test_clean_band_excludes_larger_robust_gain_if_clean_cost_exceeds_tolerance():
    rows = [row(.99,.7,10),row(.988,.8,8),row(.98,1.,2)]
    choice = choose_robust_policy(rows)
    assert choice['chosen_index'] == 1 and choice['eligible_indices'] == [0,1]
    assert choice['best_clean'] == .99


def test_robust_ties_prefer_clean_then_actual_size_then_fixed_configuration():
    rows = [row(.99,.8,10,0),row(.99,.8,8,2),row(.988,.8,3,0),row(.99,.8,8,1)]
    assert choose_robust_policy(rows)['chosen_index'] == 3
    with pytest.raises(ValueError): choose_robust_policy([])
    with pytest.raises(ValueError): choose_robust_policy(rows,clean_tolerance=-.1)
