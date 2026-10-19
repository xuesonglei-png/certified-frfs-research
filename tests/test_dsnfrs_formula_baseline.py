"""Target-class direction, strict supports, noise skipping and scalar replay."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from dsnfrs_formula_baseline import DirectedSoftNeighborhood


def model(A, y, **kwargs):
    options = dict(k=3, beta=1.6, variance_ddof=0, zero_variance_lambda=1e-8, singleton_variance='zero')
    return DirectedSoftNeighborhood(A, y, **{**options, **kwargs})


def scalar(A, y, fs, k, beta, ddof, lam):
    variances = {c: [sum((A[j,a]-np.mean(A[y == c,a]))**2 for j in np.flatnonzero(y == c))/(sum(y == c)-ddof)
                     for a in range(A.shape[1])] for c in np.unique(y)}
    D = np.array([[sum((A[i,a]-A[j,a])**2/(variances[y[j]][a] or lam) for a in fs)
                   for j in range(len(A))] for i in range(len(A))])
    lower, witnesses, ranks = [], [], []
    for i in range(len(A)):
        enemies = sorted((j for j in range(len(A)) if y[j] != y[i]), key=lambda j: (D[i,j], j))[:k]
        counts = [sum(y[z] == y[i] and D[i,z] < D[i,j] for z in range(len(A))) for j in enemies]
        t = max(range(len(enemies)), key=lambda t: (counts[t]-beta*t, -t))
        witnesses.append(enemies[t]); ranks.append(t+1); lower.append(1-np.exp(-D[i,enemies[t]]))
    return D, np.array(lower), witnesses, ranks


def test_target_variance_direction_and_strict_zero_distance_supports():
    A = np.array([[0.], [.2], [.7], [1.]]); y = np.array([0,0,1,1]); m = model(A,y)
    D = m.distances([0])
    assert D[0,2] == pytest.approx(.7**2/.0225)
    assert D[2,0] == pytest.approx(.7**2/.01)
    assert D[0,2] != pytest.approx(D[2,0])
    empty = m.profile(m.distances([]))
    assert empty['dependency'] == 0 and not np.any(empty['candidate_support_counts'])
    assert empty['witness_enemy_ranks'] == [1]*4
    duplicate = model(np.zeros((4,1)), y).profile(np.zeros((4,4)))
    assert duplicate['candidate_support_counts'] == [[0,0]]*4


def test_soft_neighbor_can_skip_noisy_nearest_enemy_and_large_penalty_reverts():
    A = np.array([0., .1, .2, .3, .4, .05, .9, 1., .99, .98])[:,None]; y = np.repeat([0,1],5)
    low = model(A,y).profile(model(A,y).distances([0]))
    high_model = model(A,y,beta=6); high = high_model.profile(high_model.distances([0]))
    assert low['witness_enemy_ranks'][0] > 1
    assert high['witness_enemy_ranks'] == [1]*len(A)
    assert low['own_class_lower'][0] > high['own_class_lower'][0]


def test_scalar_formula_replay_and_native_versus_fixed_count_stopping():
    A = np.random.default_rng(531).uniform(0,1,(18,4)); y=np.repeat([0,1,2],6)
    for ddof in (0,1):
        m=model(A,y,variance_ddof=ddof,k=5)
        for fs in ([],[0],[1,3],list(range(4))):
            D, lower, witnesses, ranks = scalar(A,y,fs,5,1.6,ddof,1e-8)
            np.testing.assert_allclose(D,m.distances(fs),atol=1e-12)
            actual=m.profile(m.distances(fs))
            np.testing.assert_allclose(lower,actual['own_class_lower'],atol=1e-12)
            assert witnesses==actual['witness_indices'] and ranks==actual['witness_enemy_ranks']
        chosen, report=m.forward(delta=1)
        assert chosen==[] and report['mode']=='native_gain_stopping'
        chosen, report=m.forward(delta=1,requested_count=3)
        assert len(chosen)==3 and report['mode']=='fixed_count_adaptation'
        for row in report['trace']:
            scores={a:scalar(A,y,sorted(row['selected_before']+[a]),5,1.6,ddof,1e-8)[1].mean()
                    for a in range(4) if a not in row['selected_before']}
            assert row['chosen_feature']==min(scores,key=lambda a:(-scores[a],a))
    with pytest.raises(ValueError): model(A,y,beta=.5)
    with pytest.raises(ValueError): model(A[:1],y[:1])
