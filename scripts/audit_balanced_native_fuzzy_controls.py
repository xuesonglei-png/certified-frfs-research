"""Independent class-balanced fuzzy fits and all native/budget choices."""
import numpy as np
from sklearn.linear_model import LogisticRegression
import audit_common_fuzzy_classifier as original
from certified_frfs.linear_retention import linear_parameters


def independent_fit(A, y, fs, C):
    classes = np.unique(y)
    W = np.zeros((len(classes), A.shape[1], 3))
    if fs:
        model = LogisticRegression(C=C, max_iter=2000, random_state=0,
                                   class_weight='balanced').fit(
            original.memberships(A[:, fs]).reshape(len(A), -1), y)
        coef, b, classes = linear_parameters(model)
        W[:, fs] = coef.reshape(len(classes), len(fs), 3)
    else:
        b = np.zeros(len(classes))
    return {'coef': W.tolist(), 'intercept': b.tolist(), 'classes': classes.tolist(),
            'class_balanced': True}


if __name__ == '__main__':
    original.audit.FOLDER = original.audit.ROOT/'results/balanced-native-fuzzy-controls'
    original.audit.independent_fit = independent_fit
    original.audit.score = original.score
    original.audit.main()
