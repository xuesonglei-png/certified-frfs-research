import numpy as np
from sklearn.svm import SVC
from certified_frfs.crossfit_kernel_projection import CrossfitKernelProjection,certified_kernel_swaps,class_scores

def test_precomputed_imputation_predicts_identically_to_independent_raw_RBF_teacher():
    rng = np.random.default_rng(643); raw = rng.uniform(size=(33,4)); y = np.arange(33)%3
    risk = CrossfitKernelProjection(raw,y); fs = [0,2]; reported = risk(fs); errors,mse = [],[]
    for fold in risk.folds:
        model = SVC(C=1.,kernel='rbf',gamma=fold['gamma'],class_weight='balanced',decision_function_shape='ovr').fit(fold['A'],fold['train'])
        Q = fold['B'].copy(); Q[:,[1,3]] = fold['center'][[1,3]]
        prediction = model.predict(Q); values = model.decision_function(Q); values -= values.mean(axis=1,keepdims=True)
        errors.append(float(fold['balance']@(prediction != fold['truth'])))
        mse.append(float(fold['balance']@np.mean((values-fold['original_scores'])**2,axis=1)))
    np.testing.assert_allclose(reported['fold_balanced_error'],errors,atol=1e-12)
    np.testing.assert_allclose(reported['fold_score_mse'],mse,atol=1e-10)
    assert risk(range(4))['score_mse'] < 1e-10

def test_same_count_swaps_keep_original_feasibility_without_query_interface():
    class Risk:
        d,C,seed = 4,1.,20261020
        def __call__(self,fs):
            return {'balanced_error':1-(sum([.01,.4,.3,.1][a] for a in fs)), 'score_mse':0.}
    def candidate(fs): return {'features':fs,'certificate':{'feasible':0 in fs}}
    result = certified_kernel_swaps([0,3],Risk(),candidate)
    assert result['features'] == [0,1] and result['candidate']['certificate']['feasible']
    assert result['final_risk']['balanced_error'] < result['initial_risk']['balanced_error']
