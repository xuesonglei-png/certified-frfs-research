"""Rebuild deletion ranking; replay raw certificates and deployment selection."""
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.predictive_deletion_support import predictive_supports
from certified_frfs.feasible_transport import weight_certificate
from audit_triangular_classifier_diagnostic import manual_margins
from run_common_fuzzy_classifier import score

FOLDER = ROOT/'results/predictive-deletion-pilot'


def main():
    threadpool_limits(1)
    summary = json.loads((FOLDER/'complete.json').read_text())
    counts = {'splits': 0, 'rebuilt_rankings': 0, 'candidates': 0,
              'margin_rejected': 0, 'repaired_to_full': 0, 'selected_full': 0,
              'selected_compact': 0, 'selected_repaired_or_fallback_score_mismatch': 0}
    rows = []
    for row in summary['rows']:
        name = row['dataset']
        saved = json.loads((FOLDER/(name+'.json')).read_text())
        X, y = load_dataset(name)
        for rel, digest in saved['fingerprints'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
        for split in saved['splits']:
            ri, vi = np.array(split['reference_indices']), np.array(split['validation_indices'])
            assert set(ri).isdisjoint(vi)
            A, B = preprocess_fit(X[ri], X[vi])
            candidates, trace = predictive_supports(X[ri], y[ri], C=split['teacher_C'])
            assert trace == split['trace'] and candidates == split['generated_candidates']
            counts['rebuilt_rankings'] += 1
            teacher = split['teacher']
            original, _ = manual_margins(A, y[ri], np.array(teacher['coef']),
                np.array(teacher['intercept']), np.array(teacher['classes']))
            np.testing.assert_allclose(original, split['original_margins'], atol=1e-10)
            protected = original > 1e-10
            for option in split['options']:
                fs = option['candidate']['features']
                v = np.array(option['candidate']['weights'])[fs]
                assert weight_certificate(A, y[ri], fs, np.ones(X.shape[1])/np.sqrt(X.shape[1]), v)['feasible']
                fit = option['fit']
                margins, _ = manual_margins(A, y[ri], np.array(fit['coef']),
                    np.array(fit['intercept']), np.array(fit['classes']))
                assert np.all(margins[protected] >= .5*original[protected]-1e-8)
                assert np.all(margins[protected] > 1e-10)
                assert np.all(np.array(fit['coef'])[:, np.setdiff1d(np.arange(X.shape[1]), fs)] == 0)
                if 'target_fit' in option:
                    target = option['target_fit']
                    target_margin, _ = manual_margins(A, y[ri], np.array(target['coef']),
                        np.array(target['intercept']), np.array(target['classes']))
                    np.testing.assert_allclose(target_margin, option['target_margins'], atol=1e-10)
                    accepted = bool(np.all(target_margin[protected] >= .5*original[protected]-1e-8)
                                    and np.all(target_margin[protected] > 1e-10))
                    assert accepted == option['margin_passed']
                    counts['margin_rejected'] += not accepted
                    counts['repaired_to_full'] += len(option['requested_repaired_candidate']['features']) == X.shape[1]
                counts['candidates'] += 1
            options = split['options']
            best = max(r['inner_balanced_accuracy'] for r in options)
            selected = min((i for i,r in enumerate(options) if r['inner_balanced_accuracy'] >= best-.005-1e-12),
                key=lambda i:(len(options[i]['candidate']['features']), -options[i]['inner_balanced_accuracy'], i))
            assert selected == split['chosen_index']
            option = options[selected]
            counts['selected_full'] += len(option['candidate']['features']) == X.shape[1]
            counts['selected_compact'] += len(option['candidate']['features']) < X.shape[1]
            counts['selected_repaired_or_fallback_score_mismatch'] += option['candidate']['features'] != option['requested_features']
            np.testing.assert_allclose(score(B, y[vi], option['fit']), split['clean'], atol=1e-12)
            np.testing.assert_allclose(score(B, y[vi], teacher), split['full_clean'], atol=1e-12)
            counts['splits'] += 1
        rows.append({'dataset': name, 'split_count': len(saved['splits'])})
        print(name, 'deletion ranking and exact certificates replayed', flush=True)
    assert counts['splits'] == 44
    counts = {key: int(value) for key,value in counts.items()}
    counts['scope'] = 'ranking_refit_certificate_prediction_replay_diagnostic_not_unbiased_selection'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    print(counts)


if __name__ == '__main__':
    main()
