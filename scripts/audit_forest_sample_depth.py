"""Independently rebuild all270 sample-depth models; no re-selection."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT, digest
from diagnose_forest_sample_depth import FOLDER, FRACTIONS
from screen_multiknot_forest import METRICS
from audit_multiknot_forest_screen import independent_scores, independent_lp
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.fuzzy_interaction_tree import interaction_tree
from certified_frfs.multiknot_forest import fit_multiknot_forest, box_margins
from certified_frfs.forest_fuzzy_classifier import forest_box_margins
from review_box_iterations import paired
from audit_forest_compact_transfer import equivalent


if __name__ == '__main__':
    threadpool_limits(1)
    summary = json.loads((FOLDER/'complete.json').read_text())
    rows, counts, hashes = [], Counter(), {}
    maxima = {'fresh_score_error': 0., 'legacy_margin_error': 0., 'LP_error': 0.}
    for expected in summary['rows']:
        name = expected['dataset']
        artifact = FOLDER/(name+'.json')
        saved = json.loads(artifact.read_text())
        assert saved['complete'] and saved['development_query_labels_scored'] and not saved['query_labels_used_for_selection']
        for rel, sha in saved['fingerprints'].items():
            assert digest(ROOT/rel) == sha, rel
        hashes[name] = digest(artifact)
        X, y = load_dataset(name)
        assert saved['raw_n'] == len(X) and saved['d'] == X.shape[1]
        assert len(saved['records']) == 9
        split_pairs = list(StratifiedKFold(3, shuffle=True, random_state=101).split(X, y))
        evaluations = {}
        for outer, (train, test) in enumerate(split_pairs):
            rng = np.random.default_rng(20261031+outer)
            orders = [rng.permutation(train[y[train] == c]) for c in np.unique(y[train])]
            previous = set()
            for fraction_index, fraction in enumerate(FRACTIONS):
                record = saved['records'][3*outer+fraction_index]
                assert record['outer'] == outer and record['fraction'] == fraction
                assert record['outer_train_indices'] == train.tolist() and record['validation_indices'] == test.tolist()
                selected = np.sort(np.concatenate([order[:max(1, int(np.ceil(fraction*len(order))))] for order in orders]))
                assert record['reference_indices'] == selected.tolist()
                assert previous.issubset(selected) and set(selected).issubset(train) and set(selected).isdisjoint(test)
                previous = set(selected)
                if fraction == 1.:
                    np.testing.assert_array_equal(selected, train)
                A, B = preprocess_fit(X[selected], X[test])
                equivalent(record['tree'], interaction_tree(A, y[selected]))
                metrics = {}
                for knot_count in (3, 5):
                    m = str(knot_count)
                    fit = record['models'][m]
                    fresh = fit_multiknot_forest(A, y[selected], list(range(A.shape[1])), record['tree']['edges'], knot_count=knot_count)
                    assert fit['C'] == fit['pair_penalty'] == 1. and fit['class_balanced'] and not fit['teacher_guards_enforced']
                    assert fit['features'] == list(range(A.shape[1])) and fit['edges'] == record['tree']['edges']
                    np.testing.assert_array_equal(fit['knots'], np.linspace(0, 1, knot_count))
                    for key in ('optimizer_iterations', 'optimizer_converged', 'design_columns'):
                        assert fit[key] == fresh[key]
                    value, freshvalue = independent_scores(B, fit), independent_scores(B, fresh)
                    maxima['fresh_score_error'] = max(maxima['fresh_score_error'], float(np.max(abs(value-freshvalue))))
                    np.testing.assert_allclose(value, freshvalue, atol=1e-8)
                    classes = np.array(fit['classes'])
                    truth = y[test]
                    own = np.searchsorted(classes, truth)
                    pred = classes[value.argmax(axis=1)]
                    oracle = box_margins(B, truth, fit, eps=.01, witnesses=True)
                    pair = oracle['pair_margins'].copy()
                    pair[np.arange(len(B)), own] = 0.
                    report = record['evaluations'][m]
                    assert report['predictions'] == pred.tolist()
                    np.testing.assert_allclose(report['scores'], value, atol=1e-10)
                    np.testing.assert_allclose(report['pair_margins'], pair, atol=1e-10)
                    witness = independent_scores(oracle['witness_raw'], fit)
                    np.testing.assert_allclose(witness[np.arange(len(B)), own]-witness[np.arange(len(B)), oracle['rivals']], oracle['margins'], atol=1e-10)
                    if knot_count == 3:
                        legacy = forest_box_margins(B, truth, fit['coef'], fit['intercept'], classes, fit['edges'],
                                                   np.array(fit['pair_coef']).reshape(len(classes), len(fit['edges']), 3, 3), eps=.01)
                        np.testing.assert_allclose(legacy, oracle['margins'], atol=1e-10)
                        maxima['legacy_margin_error'] = max(maxima['legacy_margin_error'], float(np.max(abs(legacy-oracle['margins']))))
                        counts['legacy_margin_replays'] += len(B)
                    loss = logsumexp(value, axis=1)-value[np.arange(len(B)), own]
                    def balanced(array):
                        return float(np.mean([np.mean(array[truth == c]) for c in np.unique(truth)]))
                    metrics[m] = {'clean_BA': balanced(pred == truth), 'certified_BA': balanced(oracle['margins'] > 1e-10),
                                  'nominal_logloss': balanced(loss)}
                    equivalent(metrics[m], report['metrics'])
                    if outer == 0 and fraction == 1.:
                        for r in range(len(classes)):
                            if r == own[0]:
                                continue
                            lp = independent_lp(B[0], own[0], r, fit)
                            error = float(abs(lp-pair[0, r]))
                            np.testing.assert_allclose(lp, pair[0, r], atol=1e-7)
                            maxima['LP_error'] = max(maxima['LP_error'], error)
                            counts['LP_probes'] += 1
                    counts['models'] += 1
                    counts['fresh_training_fits'] += 1
                    counts['optimizer_caps'] += not fresh['optimizer_converged']
                    counts['score_replays'] += 1
                    counts['witness_replays'] += len(B)
                evaluations[(outer, str(fraction))] = metrics
        metrics = {str(f): {m: {k: float(np.mean([evaluations[(o, str(f))][m][k] for o in range(3)]))
                               for k in METRICS} for m in ('3', '5')} for f in FRACTIONS}
        equivalent(metrics, expected['metrics'])
        gain = {f: v['5']['clean_BA']-v['3']['clean_BA'] for f, v in metrics.items()}
        equivalent(gain, expected['capacity_gain_by_fraction'])
        interaction = gain['1.0']-gain['0.3']
        equivalent(interaction, expected['capacity_gain_full_minus_small'])
        rows.append({'dataset': name, 'metrics': metrics, 'gain': gain, 'interaction': interaction})
        counts['datasets'] += 1
        print(name, 'all18 models/budgets/graphs/scores/witnesses audited', flush=True)
    for k, v in summary['counts'].items():
        assert counts[k] == v
    means = {str(f): {m: {k: float(np.mean([r['metrics'][str(f)][m][k] for r in rows]))
                         for k in METRICS} for m in ('3', '5')} for f in FRACTIONS}
    equivalent(means, summary['means'])
    descriptions = {str(f): paired([r['gain'][str(f)] for r in rows]) for f in FRACTIONS}
    equivalent(descriptions, summary['capacity_gain_by_fraction'])
    primary = paired([r['interaction'] for r in rows])
    equivalent(primary, summary['primary_sample_depth_interaction'])
    assert not summary['goal_achieved'] and not summary['support_selectors_evaluated']
    result = {'passed': True, 'counts': dict(counts), 'precision_maxima': maxima, 'artifact_hashes': hashes,
              'summary_sha256': digest(FOLDER/'complete.json'), 'auditor_sha256': digest(Path(__file__)),
              'goal_achieved': False, 'scope': 'all270_fresh_models_exposed_diagnostic_not_SOTA_or_old_gate_reversal'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result, flush=True)
