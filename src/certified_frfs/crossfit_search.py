"""Bounded subset search evaluated on internal validation queries.

Internal CV is used adaptively for search, not an unbiased risk estimator.
Outer validation and test splits must remain separate from this procedure.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from .baselines import mi_scores, relieff_scores
from .query_certificate import certify_queries


def select_crossfit_queries(X, y, *, rank='relief', mode='robust', penalty=.02,
                           budget=24, seed=101, constraint_tolerance=None):
    X, y = np.asarray(X, float), np.asarray(y)
    if mode not in ['robust', 'nominal', 'accuracy'] or rank not in ['mi', 'relief']:
        raise ValueError('invalid search mode or rank')
    if X.ndim != 2 or len(X) != len(y) or budget < 7 or penalty < 0:
        raise ValueError('invalid training data or search budget')
    if constraint_tolerance is not None and not 0 <= constraint_tolerance <= 1:
        raise ValueError('invalid empirical prediction tolerance')
    counts = np.unique(y, return_counts=True)[1]
    folds = min(2, counts.min())
    if folds < 2:
        raise ValueError('internal query evaluation needs two objects per class')
    splits = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed+17).split(X, y))
    references, rank_sum, preprocessing = [], np.zeros(X.shape[1]), []
    d = X.shape[1]
    for tr, va in splits:
        imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        scaler = MinMaxScaler(clip=True)
        A = scaler.fit_transform(imputer.fit_transform(X[tr]))
        B = scaler.transform(imputer.transform(X[va]))
        relevance = (mi_scores(A, y[tr], random_state=seed) if rank == 'mi'
                     else relieff_scores(A, y[tr], random_state=seed))
        ordering = np.argsort(-relevance, kind='stable')
        rank_sum[ordering] += np.arange(d, 0, -1)/d
        references.append((A, y[tr], B, y[va]))
        preprocessing.append({'reference_indices': tr.tolist(), 'query_indices': va.tolist(),
                              'imputer_statistics': imputer.statistics_.tolist(),
                              'data_min': scaler.data_min_.tolist(), 'data_max': scaler.data_max_.tolist()})
    ranking = np.argsort(-rank_sum, kind='stable').tolist()
    cache, trace = {}, []

    def evaluate(features):
        key = tuple(sorted(features))
        if key in cache:
            return cache[key]
        scores, diagnostics = [], []
        for A, labels, B, truth in references:
            model = KNeighborsClassifier(n_neighbors=1, metric='manhattan', algorithm='brute').fit(A[:, key], labels)
            pred = model.predict(B[:, key])
            correct = pred == truth
            if mode == 'accuracy':
                protected = np.ones(len(truth), dtype=bool)
            else:
                cert = certify_queries(A, labels, B, pred, key, eps=.01 if mode=='robust' else 0.)
                protected = np.array(cert['certified'])
            per_class = [{'label': int(c), 'accuracy': float(correct[truth==c].mean()),
                          'certified_correct': float((correct & protected)[truth==c].mean()),
                          'certified_wrong': float((~correct & protected)[truth==c].mean())}
                         for c in np.unique(truth)]
            ba = np.mean([r['accuracy'] for r in per_class])
            cc = np.mean([r['certified_correct'] for r in per_class])
            objective = ba if mode=='accuracy' or constraint_tolerance is not None else .75*ba+.25*cc
            scores.append(objective)
            diagnostics.append({'balanced_accuracy': float(ba), 'balanced_certified_correct': float(cc),
                                'classes': per_class,
                                'certificate_evaluated': mode!='accuracy'})
        row = {'features': list(key), 'objective': float(np.mean(scores)-penalty*len(key)/d),
               'fold_diagnostics': diagnostics}
        cache[key] = row
        trace.append(row)
        return row

    def value(row):
        return (round(row['objective'],12), -len(row['features']), tuple(-a for a in row['features']))

    for fraction in [.1, .2, .35, .5, .65, .8, 1.]:
        evaluate(ranking[:max(1,int(np.ceil(fraction*d)))])
    current = max(trace,key=value)
    initial = current['features'].copy()
    while len(cache)<budget:
        selected = current['features']
        present = set(selected)
        retained = [a for a in reversed(ranking) if a in present][:4]
        omitted = [a for a in ranking if a not in present][:4]
        moves = ([[a for a in selected if a!=old] for old in retained] if len(selected)>1 else [])
        moves += [selected+[new] for new in omitted]
        moves += [[a for a in selected if a!=old]+[new] for old in retained for new in omitted]
        best = current
        for move in moves:
            if tuple(sorted(move)) not in cache and len(cache)>=budget:
                break
            candidate = evaluate(move)
            if value(candidate)>value(best):
                best=candidate
        if best is current:
            break
        current=best
    selection = None
    if constraint_tolerance is not None:
        def predictive_score(row):
            return float(np.mean([f['balanced_accuracy'] for f in row['fold_diagnostics']]))
        full = cache[tuple(range(d))]
        reference_score = predictive_score(full)
        threshold = reference_score-constraint_tolerance
        eligible = [row for row in trace if predictive_score(row) >= threshold-1e-12]
        def constrained_value(row):
            if mode == 'accuracy':
                utility = -len(row['features'])/d
            else:
                cc = np.mean([f['balanced_certified_correct'] for f in row['fold_diagnostics']])
                wrong = np.mean([np.mean([c['certified_wrong'] for c in f['classes']])
                                 for f in row['fold_diagnostics']])
                utility = cc-wrong-penalty*len(row['features'])/d
            return (round(float(utility),12), round(predictive_score(row),12),
                    -len(row['features']), tuple(-a for a in row['features']))
        current = max(eligible,key=constrained_value)
        selection = {'reference_score':reference_score, 'threshold':threshold,
                     'tolerance':constraint_tolerance, 'eligible':len(eligible),
                     'selected_predictive_score':predictive_score(current),
                     'selected_utility':constrained_value(current)[0],
                     'full_fallback':len(current['features'])==d,
                     'scope':'empirical_internal_CV_constraint_not_test_accuracy_bound'}
    return {'features':current['features'], 'initial_features':initial,
            'objective':current['objective'], 'ranking':ranking, 'trace':trace,
            'evaluations':len(cache), 'budget':budget, 'mode':mode,
            'preprocessing':preprocessing,
            'constraint_selection':selection,
            'scope':'adaptive_internal_CV_selection_not_unbiased_test_risk_or_certificate'}
