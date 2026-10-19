import numpy as np
from scipy.special import logsumexp
from certified_frfs.forest_robust_logloss import all_pair_box_witnesses,fit_robust_forest_logloss
from certified_frfs.orthogonal_forest_logistic import fit_orthogonal_forest,orthogonal_scores
from certified_frfs.forest_fuzzy_classifier import forest_scores


def test_binary_loss_is_exact_and_opposed_multiclass_rivals_make_envelope_strict():
    X = np.array([[.5]]); y = np.array([0]); W = [[[0.,0.,0.]],[[-.5,0.,.5]],[[.5,0.,-.5]]]
    fit = {'classes':[0,1,2],'coef':W,'intercept':[0.,0.,0.],'edges':[],'pair_coef':np.zeros((3,0,3,3)).tolist()}
    oracle = all_pair_box_witnesses(X,y,fit,eps=.5)
    states = np.linspace(0,1,1001)[:,None]; scores = forest_scores(states,W,[0.,0.,0.],[],np.zeros((3,0,3,3)))
    worst = np.max(logsumexp(scores,axis=1)-scores[:,0])
    assert oracle['envelope_losses'][0] > worst+.1
    np.testing.assert_allclose(oracle['pair_witness_raw'][0,1:,0],[1.,0.],atol=1e-12)
    binary = {**fit,'classes':[0,1],'coef':W[:2],'intercept':[0.,0.],'pair_coef':np.zeros((2,0,3,3)).tolist()}
    exact = all_pair_box_witnesses(X,y,binary,eps=.5)
    np.testing.assert_allclose(exact['envelope_losses'][0],np.logaddexp(0,.5),atol=1e-12)


def test_zero_radius_matches_ordinary_class_balanced_penalized_objective():
    rng = np.random.default_rng(117); X = rng.uniform(size=(30,2)); y = np.arange(30)%3
    robust = fit_robust_forest_logloss(X,y,[0,1],[[0,1]],eps=0.)
    assert robust['usable'] and robust['status'] == 'separated' and len(robust['trace']) == 1
    ordinary = fit_orthogonal_forest(X,y,[0,1],[[0,1]],C=1.,pair_penalty=1.)
    score = orthogonal_scores(X,ordinary); loss = np.mean(logsumexp(score,axis=1)-score[np.arange(30),y])
    W,V = np.array(ordinary['coef']),np.array(ordinary['pair_coef']); W -= W.mean(axis=0,keepdims=True); V -= V.mean(axis=0,keepdims=True)
    objective = loss+(np.sum(W*W)+np.sum(V*V))/(2*len(X))
    np.testing.assert_allclose(robust['exact_envelope_objective'],objective,atol=2e-7)


def test_binary_positive_radius_separates_all_true_label_contrasts_without_teacher_guards():
    X = np.array([[.1],[.2],[.3],[.7],[.8],[.9]]); y = np.array([0,0,0,1,1,1])
    fit = fit_robust_forest_logloss(X,y,[0],[],eps=.1)
    assert fit['usable'] and len(fit['cuts']) > 0 and not fit['teacher_guards_enforced']
    oracle = all_pair_box_witnesses(X,y,fit,eps=.1)
    assert np.max(-oracle['pair_margins']-np.array(fit['upper'])) <= fit['separation_tol']+1e-10
    assert fit['exact_envelope_objective']-fit['relaxation_objective'] <= 3e-7
