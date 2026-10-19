import numpy as np
import pytest
from sklearn.svm import SVC
from certified_frfs.crossfit_rbf_refit import CrossfitRBFRefit, certified_refit_swaps, foldwise_improves


@pytest.mark.parametrize('classes',[2,3])
def test_refitted_kernel_loss_matches_raw_support_svm_and_recomputes_gamma(classes):
    raw = np.random.default_rng(479).normal(size=(36,5)); y = np.arange(36)%classes
    risk = CrossfitRBFRefit(raw,y); fs = [1,3]; result = risk(fs)
    for fold,fit,error,hinge in zip(risk.folds,result['fold_fits'],result['fold_balanced_error'],result['fold_balanced_hinge']):
        A,B = fold['A'][:,fs],fold['B'][:,fs]
        model = SVC(C=1.,kernel='rbf',gamma='scale',class_weight='balanced',decision_function_shape='ovr').fit(A,fold['train'])
        assert fit['gamma'] == model._gamma
        pred = model.predict(B); assert fit['predictions'] == pred.tolist()
        np.testing.assert_allclose(error,fold['balance']@(pred != fold['truth']),atol=1e-12)
        score = model.decision_function(B); index = np.searchsorted(model.classes_,fold['truth'])
        if classes == 2: margin = (2*index-1)*score
        else:
            margin = np.array([score[i,c]-max(score[i,k] for k in range(classes) if k != c) for i,c in enumerate(index)])
        np.testing.assert_allclose(hinge,fold['balance']@np.maximum(1-margin,0),atol=1e-8)
    assert risk(fs) is result


def test_foldwise_rule_rejects_a_mean_gain_that_harms_one_fold():
    base = {'balanced_error':.3,'fold_balanced_error':[.3,.3,.3]}
    assert not foldwise_improves({'balanced_error':.2,'fold_balanced_error':[.1,.1,.4]},base)
    assert foldwise_improves({'balanced_error':.2,'fold_balanced_error':[.1,.2,.3]},base)


def test_refit_search_skips_infeasible_better_swap_and_preserves_count():
    class Risk:
        d,C,seed = 4,1.,20261020
        def __call__(self,fs):
            error = 1-sum([.05,.4,.3,.1][a] for a in fs)
            return {'balanced_error':error,'balanced_hinge':error,'fold_balanced_error':[error]*3}
    def candidate(fs): return {'features':fs,'certificate':{'feasible':0 in fs}}
    result = certified_refit_swaps([0,3],Risk(),candidate)
    assert result['features'] == [0,1]
    assert len(result['features']) == 2 and any(not a['accepted'] for a in result['attempts'])
