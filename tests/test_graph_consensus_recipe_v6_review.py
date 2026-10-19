"""Hand-constructed independent checks of the frozen v6 aggregation rule."""
import numpy as np
import pytest
from graph_consensus_recipe_v6 import consensus_ranking


def test_native_frequency_precedes_borda_and_feature_id_breaks_exact_tie():
    registry=[({}, {'native':{'features':[0]},'ranking':[3,2,1,0]}),
              ({}, {'native':{'features':[0]},'ranking':[1,2,3,0]})]
    ranking,record=consensus_ranking(registry,4)
    assert ranking==[0,1,2,3]
    np.testing.assert_array_equal(record['native_frequency'],[1.,0.,0.,0.])
    np.testing.assert_array_equal(record['mean_rank'],[3.,1.,1.,1.])


def test_equal_config_weight_is_invariant_to_order_and_uniform_registry_replication():
    registry=[({'a':1},{'native':{'features':[0,2]},'ranking':[2,0,3,1]}),
              ({'a':2},{'native':{'features':[1,2]},'ranking':[1,2,0,3]}),
              ({'a':3},{'native':{'features':[0,1]},'ranking':[0,1,3,2]})]
    expected,record=consensus_ranking(registry,4)
    actual,reordered=consensus_ranking(registry[::-1],4)
    repeated,repeat_record=consensus_ranking(registry*2,4)
    assert expected==actual==repeated
    assert record['native_frequency']==reordered['native_frequency']==repeat_record['native_frequency']
    assert record['mean_rank']==reordered['mean_rank']==repeat_record['mean_rank']


@pytest.mark.parametrize('ranking',[[0,0,2],[0,1],[0,1,3]])
def test_invalid_graph_permutation_is_rejected(ranking):
    with pytest.raises(ValueError,match='invalid graph rank'):
        consensus_ranking([({}, {'native':{'features':[0]},'ranking':ranking})],3)


def test_empty_training_graph_registry_is_rejected():
    with pytest.raises(ValueError,match='empty graph registry'):
        consensus_ranking([],4)
