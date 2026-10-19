"""Separate teacher shift, unconstrained controls and reduction effects."""
import json
import numpy as np
from scipy.special import logsumexp
from review_box_iterations import ROOT, paired
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.triangular_classifier import triangular_memberships
from audit_triangular_classifier_diagnostic import manual_margins


def objective(A, y, W, b, classes, C):
    own = np.argmax(y[:, None] == classes[None, :], axis=1)
    weights = len(y)/(len(classes)*np.bincount(own, minlength=len(classes))[own])
    scores = np.einsum('ndk,qdk->nq', triangular_memberships(A), W)+b
    loss = weights@(logsumexp(scores, axis=1)-scores[np.arange(len(y)), own])
    penalty = W[1] if len(classes) == 2 else W-W.mean(axis=0, keepdims=True)
    return float(loss+np.sum(penalty**2)/(2*C))


def main():
    folder = ROOT/'results/nested-balanced-teacher-backward'
    proposed = json.loads((folder/'complete.json').read_text())
    native = json.loads((folder.parent/'balanced-native-fuzzy-controls/complete.json').read_text())
    controls = {r['dataset']: r['methods'] for r in native['rows']}
    old = {r['dataset']: r for r in json.loads((folder.parent/'nested-joint-backward/complete.json').read_text())['rows']}
    rows, shifts = [], []
    for row in proposed['rows']:
        name = row['dataset']
        saved = json.loads((folder/(name+'.json')).read_text())
        previous = json.loads((folder.parent/'nested-joint-backward'/(name+'.json')).read_text())
        X, y = load_dataset(name)
        for split, prior in zip(saved['splits'], previous['splits']):
            ri = np.array(split['reference_indices'])
            A, _ = preprocess_fit(X[ri], X[ri])
            stage = split['stage']
            classes = np.array(stage['teacher_classes'])
            W, b = np.array(stage['teacher_coef']), np.array(stage['teacher_intercept'])
            margins, _ = manual_margins(A, y[ri], W, b, classes)
            teacher_obj = objective(A, y[ri], W, b, classes, stage['teacher_C'])
            full = stage['policies']['1.0']['fit']
            full_obj = objective(A, y[ri], np.array(full['coef']), np.array(full['intercept']), classes, stage['teacher_C'])
            # Compare masks at the SAME C, not different selected teachers.
            from sklearn.linear_model import LogisticRegression
            from certified_frfs.linear_retention import linear_parameters
            teacher = LogisticRegression(C=stage['teacher_C'], max_iter=2000, random_state=0).fit(
                triangular_memberships(A).reshape(len(A), -1), y[ri])
            oldW, oldb, oldclasses = linear_parameters(teacher)
            oldm, _ = manual_margins(A, y[ri], oldW.reshape(W.shape), oldb, oldclasses)
            shifts.append({'dataset': name, 'C': stage['teacher_C'],
                          'balanced_protected': int(sum(margins > 1e-10)),
                          'ordinary_protected': int(sum(oldm > 1e-10)),
                          'gained': int(sum((margins > 1e-10)&(oldm <= 1e-10))),
                          'lost': int(sum((margins <= 1e-10)&(oldm > 1e-10))),
                          'teacher_objective': teacher_obj, 'full_objective': full_obj,
                          'full_objective_minus_teacher': full_obj-teacher_obj})
        rows.append({'dataset': name, 'clean': row['clean'], 'full': row['full'], 'removed': row['removed'],
                     'previous_ordinary_teacher': old[name]['clean'],
                     'controls': {k: controls[name][k] for k in ('full', 'fgs2023_native', 'sfrip2025_native')}})
    report = {'rows': rows, 'anchor_and_objective_diagnostics': shifts,
              'mean_clean': proposed['mean_clean'], 'mean_removed': proposed['mean_removed'],
              'vs_matched_full': proposed['vs_matched_full'],
              'vs_ordinary_teacher_method': paired([r['clean']-r['previous_ordinary_teacher'] for r in rows]),
              'vs_balanced_native_controls': {k: paired([r['clean']-r['controls'][k]['clean'] for r in rows])
                  for k in ('full', 'fgs2023_native', 'sfrip2025_native')},
              'scope': 'changed_balanced_teacher_anchor_object_exposed_development_not_confirmation'}
    (folder/'teacher-alignment-review.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print({k: v for k, v in report.items() if k not in ('rows', 'anchor_and_objective_diagnostics')}, flush=True)
    print('same-C anchor gained/lost', sum(r['gained'] for r in shifts), sum(r['lost'] for r in shifts), flush=True)
    print('maximum full objective excess over teacher', max(r['full_objective_minus_teacher'] for r in shifts), flush=True)


if __name__ == '__main__':
    main()
