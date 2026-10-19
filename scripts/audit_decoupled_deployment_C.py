"""Independent hierarchical choices, original scalar margins and loss replay."""
import hashlib
import json
import numpy as np
from threadpoolctl import threadpool_limits
from review_box_iterations import ROOT
from review_balanced_teacher_backward import objective
from audit_triangular_classifier_diagnostic import manual_margins
from audit_nested_triangular import independent_score
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate

FOLDER = ROOT/'results/decoupled-deployment-C'
CS = (.1, 1., 10.)
FRACTIONS = ('0.5', '0.65', '0.8', '1.0')


def audit_stage(raw, y, query, truth, fitted, source, stored, counts, *, seed=None):
    assert fitted['teacher_C'] == source['teacher_C']
    for key in ('teacher_coef', 'teacher_intercept', 'teacher_classes'):
        assert fitted[key] == source[key]
    A, B = preprocess_fit(raw, query)
    classes = np.array(fitted['teacher_classes'])
    original, _ = manual_margins(A, y, np.array(fitted['teacher_coef']), np.array(fitted['teacher_intercept']), classes)
    protected = original > 1e-10
    full = fitted['policies']['1.0']['fit']
    queries = [B]
    if seed is not None:
        rng = np.random.default_rng(seed)
        queries += [np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1) for _ in range(3)]
    seen = set()
    for fraction, record in fitted['policies'].items():
        assert record['candidate'] == source['policies'][fraction]['candidate']
        assert record['requested_features'] == record['candidate']['features']
        assert record['fallback'] == (not record['solver_fit']['feasible'])
        fs = record['actual_features']
        fit = record['fit']
        if record['fallback']:
            assert fs == list(range(fitted['d'])) and fit == full
            counts['fallbacks'] += 1
        else:
            assert fs == record['requested_features']
            assert record['solver_fit']['class_balanced']
        identity = (tuple(fs), str(fit['coef']), str(fit['intercept']))
        if identity not in seen:
            seen.add(identity)
            W, b = np.array(fit['coef']), np.array(fit['intercept'])
            assert np.all(W[:, np.setdiff1d(np.arange(fitted['d']), fs)] == 0)
            actual, _ = manual_margins(A, y, W, b, classes)
            assert np.all(actual[protected] >= .5*original[protected]-1e-8)
            assert np.all(actual[protected] > 1e-10)
            if 'objective' in fit:
                value = objective(A, y, W, b, classes, fitted['deployment_C'])
                np.testing.assert_allclose(value, fit['objective'], rtol=1e-8, atol=1e-7)
                np.testing.assert_allclose(actual, fit['deployed_margins'], atol=1e-10)
                counts['objectives'] += 1
            candidate = record['candidate']
            target = candidate['features']
            weights = np.array(candidate['weights'])[target]
            assert weight_certificate(A, y, target, np.ones(fitted['d'])/np.sqrt(fitted['d']), weights)['feasible']
            counts['models'] += 1
        values = [independent_score(Q, truth, classes, fit) for Q in queries]
        np.testing.assert_allclose(stored[fraction]['clean'], values[0], atol=1e-12)
        assert stored[fraction]['retained'] == len(fs)/fitted['d']
        if seed is not None:
            np.testing.assert_allclose(stored[fraction]['query_noise01'], np.mean(values[1:]), atol=1e-12)
        counts['predictions'] += len(queries)
    counts['stages'] += 1


def main():
    threadpool_limits(1)
    summary = json.loads((FOLDER/'complete.json').read_text())
    counts = {'outer_splits': 0, 'inner_splits': 0, 'stages': 0, 'models': 0,
              'objectives': 0, 'predictions': 0, 'fallbacks': 0}
    for row in summary['rows']:
        name = row['dataset']
        saved = json.loads((FOLDER/(name+'.json')).read_text())
        source = json.loads((FOLDER.parent/'nested-balanced-teacher-backward'/(name+'.json')).read_text())
        X, y = load_dataset(name)
        for rel, digest in saved['fingerprints'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
        for index, (split, previous) in enumerate(zip(saved['splits'], source['splits'])):
            ri, vi = np.array(split['reference_indices']), np.array(split['validation_indices'])
            assert split['reference_indices'] == previous['reference_indices']
            assert split['validation_indices'] == previous['validation_indices'] and set(ri).isdisjoint(vi)
            raw, labels = X[ri], y[ri]
            teacher_options = [{'C': C, 'score': float(np.mean([
                r['by_C'][str(C)]['policies']['1.0']['clean'] for r in previous['inner']]))} for C in CS]
            teacher_C = min(teacher_options, key=lambda r: (-r['score'], r['C']))['C']
            assert split['teacher_C'] == teacher_C
            for inner, old in zip(split['inner'], previous['inner']):
                a, b = np.array(inner['reference_indices']), np.array(inner['validation_indices'])
                assert inner['reference_indices'] == old['reference_indices']
                assert inner['validation_indices'] == old['validation_indices'] and set(a).isdisjoint(b)
                for C, r in inner['by_C'].items():
                    assert float(C) == r['stage']['deployment_C']
                    audit_stage(raw[a], labels[a], raw[b], labels[b], r['stage'],
                        old['by_C'][str(teacher_C)]['stage'], r['metrics'], counts)
                counts['inner_splits'] += 1
            options = []
            for C in CS:
                for fraction in FRACTIONS:
                    metrics = [r['by_C'][str(C)]['metrics'][fraction] for r in split['inner']]
                    options.append({'C': C, 'fraction': fraction, 'score': float(np.mean([r['clean'] for r in metrics])),
                                    'retained': float(np.mean([r['retained'] for r in metrics]))})
            eligible = [r for r in options if r['score'] >= max(z['score'] for z in options)-.005-1e-12]
            selected = min(eligible, key=lambda r: (r['retained'], -r['score'], r['C'], float(r['fraction'])))
            full = min((r for r in options if r['fraction'] == '1.0'), key=lambda r: (-r['score'], r['C']))
            assert split['selected_config'] == selected and split['full_config'] == full
            assert split['selection'] == {'options': options, 'eligible': eligible, 'tolerance': .005}
            stored = {}
            from run_decoupled_deployment_C import metrics
            for C, fitted in split['outer_stages'].items():
                # Recompute saved-field structure, then independently replay each scalar score.
                values = metrics(raw, X[vi], y[vi], fitted, seed=144+index)
                audit_stage(raw, labels, X[vi], y[vi], fitted, previous['outer_stages'][str(teacher_C)],
                            values, counts, seed=144+index)
                stored[C] = values
            assert split['metrics'] == stored[str(selected['C'])][selected['fraction']]
            assert split['full_metrics'] == stored[str(full['C'])]['1.0']
            counts['outer_splits'] += 1
        print(name, 'teacher/deployment choices and scalar deployment audit passed', flush=True)
    assert counts['outer_splits'] == 44 and counts['inner_splits'] == 128
    counts['scope'] = 'teacher_choice_fixed_supports_all_deployment_objectives_margins_and_prediction_replay_not_solver_optimality'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    print(counts, flush=True)


if __name__ == '__main__':
    main()
