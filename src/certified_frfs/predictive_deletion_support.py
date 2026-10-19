"""Reference-only crossfit deletion loss; one-shot ranking, not subset optimum."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from .nested_triangular_stage import inner_splits
from .validated_pool import preprocess_fit
from .triangular_classifier import triangular_memberships


def predictive_supports(raw, y, *, C, fractions=(.5, .65, .8)):
    raw, y = np.asarray(raw, float), np.asarray(y)
    if raw.ndim != 2 or len(y) != len(raw) or not np.isfinite(C) or C <= 0:
        raise ValueError('invalid reference or C')
    d = raw.shape[1]
    if d < 2:
        raise ValueError('at least two raw attributes required')
    if not fractions or any(not 0 < f < 1 for f in fractions):
        raise ValueError('compact fractions required')
    folds = []
    for a, b in inner_splits(y):
        A, B = preprocess_fit(raw[a], raw[b])
        folds.append({'reference_indices': a.tolist(), 'validation_indices': b.tolist(),
                      'A': triangular_memberships(A), 'B': triangular_memberships(B),
                      'train_y': y[a], 'test_y': y[b]})

    def evaluate(fs):
        losses, accuracies = [], []
        for fold in folds:
            labels, truth = fold['train_y'], fold['test_y']
            model = LogisticRegression(C=C, max_iter=2000, random_state=0).fit(
                fold['A'][:, fs].reshape(len(labels), -1), labels)
            probability = model.predict_proba(fold['B'][:, fs].reshape(len(truth), -1))
            own = np.argmax(truth[:, None] == model.classes_[None, :], axis=1)
            per_object = -np.log(np.maximum(probability[np.arange(len(truth)), own], 1e-15))
            prediction = model.classes_[probability.argmax(axis=1)]
            losses.append(float(np.mean([per_object[truth == c].mean() for c in np.unique(truth)])))
            accuracies.append(float(np.mean([(prediction[truth == c] == c).mean() for c in np.unique(truth)])))
        return {'losses': losses, 'balanced_accuracies': accuracies,
                'mean_loss': float(np.mean(losses)), 'mean_balanced_accuracy': float(np.mean(accuracies))}

    full = evaluate(list(range(d)))
    deletion = []
    for attribute in range(d):
        values = evaluate([a for a in range(d) if a != attribute])
        deletion.append({'attribute': attribute, **values,
                         'loss_increase': values['mean_loss']-full['mean_loss']})
    ranking = sorted(range(d), key=lambda a: (-deletion[a]['loss_increase'], a))
    candidates = []
    for size in sorted({int(np.ceil(d*f)) for f in fractions}):
        if size == d:
            continue
        fs = sorted(ranking[:size])
        candidates.append({'features': fs, 'origin': 'crossfit_single_deletion_loss_ranking',
                           'crossfit_metrics': evaluate(fs)})
    trace = {'full': full, 'deletion': deletion, 'ranking': ranking,
             'folds': [{k:v for k,v in f.items() if k.endswith('_indices')} for f in folds],
             'C': float(C), 'scope': 'reference_only_single_deletion_not_sequential_or_optimal'}
    return candidates, trace
