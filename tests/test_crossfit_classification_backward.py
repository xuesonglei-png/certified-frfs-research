import numpy as np
from certified_frfs.crossfit_classification_backward import CrossfitClassificationLoss


def test_classification_cost_is_balanced_fold_error_then_MSE():
    rng = np.random.default_rng(422)
    X = rng.random((19, 4)); y = np.array([0]*12+[1]*7)
    loss = CrossfitClassificationLoss(X, y)
    fs = [0, 2]; cols = [0]+[1+3*a+m for a in fs for m in range(3)]
    P = np.eye(len(cols))*.001; P[0, 0] = 0
    errors, mse = [], []
    for fold in loss.folds:
        beta = np.linalg.solve(fold['gram'][np.ix_(cols, cols)]+P, fold['cross'][cols])
        score = fold['query'][:, cols]@beta
        own = fold['target'].argmax(axis=1); pred = score.argmax(axis=1)
        errors.append(np.mean([np.mean(pred[own == c] != c) for c in np.unique(own)]))
        mse.append(fold['weights']@np.mean((score-fold['target'])**2, axis=1))
    np.testing.assert_allclose(loss(fs), (np.mean(errors), np.mean(mse)), atol=1e-12)
    np.testing.assert_allclose(loss(fs), CrossfitClassificationLoss(X, 1-y)(fs), atol=1e-12)
