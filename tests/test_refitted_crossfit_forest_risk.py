import itertools
import numpy as np
from scipy.special import logsumexp
from certified_frfs.cached_complementarity_forest import ComplementarityForestCache
from certified_frfs.fuzzy_interaction_tree import interaction_tree
from certified_frfs.refitted_crossfit_forest_risk import RefittedCrossfitForestRisk
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.orthogonal_forest_logistic import orthogonal_scores
from certified_frfs.refitted_risk_geometry_stage import RefittedRiskGeometryContext
from certified_frfs.feasible_transport import weight_certificate


def test_cached_full_edge_information_relearns_same_support_tree_all_subsets():
    rng = np.random.default_rng(281)
    X = rng.uniform(size=(36,5)); y = np.arange(36)%3
    cache = ComplementarityForestCache(X,y)
    for count in range(1,6):
        for fs in itertools.combinations(range(5),count):
            direct = interaction_tree(X[:,fs],y)
            assert cache.edges(fs) == [sorted([fs[u],fs[v]]) for u,v in direct['edges']]


def test_refitted_risk_is_actual_local_classifier_prediction_with_fold_scaling():
    rng = np.random.default_rng(542)
    raw = rng.normal(size=(19,4)); raw[-1] *= 50; y = np.array([0]*9+[1]*9+[2])
    risk = RefittedCrossfitForestRisk(raw,y); scored = risk([0,2])
    assert scored['features'] == [0,2] and len(scored['fold_model_ids']) == 3
    values = []
    for j,fold in enumerate(risk.records):
        a,b = np.array(fold['reference_indices']),np.array(fold['validation_indices'])
        assert 18 in a and 18 not in b and set(a).isdisjoint(b)
        A,B = preprocess_fit(raw[a],raw[b]); record = risk.models[scored['fold_model_ids'][j]]
        fit = record['fit']; assert fit['features'] == [0,2]
        assert fit['edges'] == [[0,2]]
        scores = orthogonal_scores(B,fit); classes = np.array(fit['classes']); own = np.argmax(y[b,None] == classes[None],axis=1)
        balance = np.array([1/(len(np.unique(y[b]))*sum(y[b] == c)) for c in y[b]])
        value = float(balance @ (logsumexp(scores,axis=1)-scores[np.arange(len(b)),own]))
        np.testing.assert_allclose(record['balanced_logloss'],value,atol=1e-12); values.append(value)
    np.testing.assert_allclose(scored['mean_balanced_logloss'],np.mean(values),atol=1e-12)
    assert risk([2,0]) is scored


def test_actual_risk_search_retains_original_certificate_and_scores_only_feasible_candidates():
    rng = np.random.default_rng(367)
    raw = rng.uniform(size=(24,5)); y = (raw[:,0] > .5).astype(int)
    context = RefittedRiskGeometryContext(raw,y)
    stage = context.fit_policy(C=1.,pair_penalty=None,fraction=.5)
    path = stage['path']; scores = stage['risk']['scores']
    for candidate in path['trace']:
        fs = candidate['features']; weights = np.array(candidate['weights'])[fs]
        independently_checked = weight_certificate(context.A,y,fs,context.original_weights,weights)
        assert independently_checked['feasible'] and independently_checked == candidate['certificate']
    for attempt in path['attempts']:
        assert (attempt['risk_score_id'] is not None) == attempt['certificate']['feasible']
        if attempt['accepted']: assert attempt['risk_score_id'] is not None
    for ranking in path['rankings']:
        if ranking['choices']:
            first = ranking['choices'][0]
            assert path['attempts'][first['attempt_id']]['accepted']
    assert stage['fit']['features'] == stage['candidate']['features']
    assert scores[stage['candidate']['risk_score_id']]['features'] == stage['fit']['features']
