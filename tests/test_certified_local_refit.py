from certified_frfs.certified_local_refit import certified_local_refit
from certified_frfs.crossfit_rbf_refit import certified_refit_swaps


def test_exhaustive_search_finds_feasible_neighbor_beyond_top_eight():
    class Risk:
        d,C,seed = 11,1.,20261020
        def __call__(self,fs):
            error = .9 if fs == [0] else .01*fs[0]
            return {'balanced_error':error,'balanced_hinge':error,'fold_balanced_error':[error]*3}
    def candidate(fs): return {'features':fs,'certificate':{'feasible':fs in ([0],[9])}}
    truncated = certified_refit_swaps([0],Risk(),candidate)
    result = certified_local_refit([0],Risk(),candidate)
    assert truncated['features'] == [0] and result['features'] == [9]
    assert result['accepted_steps'] == 1 and result['termination'] == 'exhaustive_one_exchange_foldwise_stationary'
    final = result['stages'][-1]
    assert len(final['ranked_choices']) == 8
    assert all(not a['accepted'] for a in result['attempts'] if a['round'] == 1)


def test_exhaustive_descent_continues_past_two_rounds_without_revisiting_supports():
    class Risk:
        d,C,seed = 7,1.,20261020
        def __call__(self,fs):
            error = 1-sum(fs)/30
            return {'balanced_error':error,'balanced_hinge':error,'fold_balanced_error':[error]*3}
    def candidate(fs): return {'features':fs,'certificate':{'feasible':True}}
    result = certified_local_refit([0,1,2],Risk(),candidate)
    assert result['features'] == [4,5,6] and result['accepted_steps'] == 3
    path = [tuple(s['features']) for s in result['stages']]
    assert len(path) == len(set(path))
    assert result['stages'][-1]['foldwise_improving_candidates'] == 0
