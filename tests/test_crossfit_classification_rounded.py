import numpy as np
from certified_frfs.crossfit_classification_rounded import CrossfitClassificationLoss


def test_primary_error_has_explicit_tie_precision():
    rng = np.random.default_rng(423)
    X = rng.random((29, 4)); y = np.array([0]*18+[1]*11)
    loss = CrossfitClassificationLoss(X, y)
    for fs in ([0], [1, 2], [0, 1, 2, 3]):
        error, mse = loss(fs)
        details = loss.details[tuple(fs)]
        assert error == round(details['balanced_error'], 12)
        assert abs(error-details['balanced_error']) <= 5.1e-13
        assert mse == details['MSE']
