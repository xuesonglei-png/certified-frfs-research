"""Audit both numerical diagnoses without evaluating validation predictions."""
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT, digest
from screen_robust_logloss_classifier import FOLDER
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.forest_fuzzy_classifier import forest_scores, forest_box_margins
from certified_frfs.orthogonal_forest_logistic import CONTRAST
from audit_fuzzy_forest_capacity import lp_margin


def audit(path, fit_key):
    saved = json.loads(path.read_text())
    assert saved['complete'] and not saved['validation_scores_used'] and not saved['original_results_overwritten']
    for rel, sha in saved['fingerprints'].items():
        assert digest(ROOT/rel) == sha, rel
    original = json.loads((FOLDER/'complete.json').read_text())
    expected = set()
    for row in original['rows']:
        source = json.loads((FOLDER/(row['dataset']+'.json')).read_text())
        expected.update((row['dataset'], i) for i, f in enumerate(source['folds']) if f['ordinary_fallback'])
    identities = [(r['dataset'], r['fold']) for r in saved['rows']]
    assert len(identities) == len(set(identities)) == len(expected)
    assert set(identities) == expected
    pairs = cuts = probes = recovered = 0
    max_error = 0.
    for row in saved['rows']:
        source = json.loads((FOLDER/(row['dataset']+'.json')).read_text())
        for rel, sha in source['fingerprints'].items():
            assert digest(ROOT/rel) == sha, rel
        fold = source['folds'][row['fold']]
        assert fold['ordinary_fallback'] and row['original_status'] == fold['fit']['status']
        fit = row[fit_key]
        if not fit['usable']:
            assert fit['status'] != 'separated'
            print(path.stem, row['dataset'], row['fold'], 'retained failure', fit['status'], flush=True)
            continue
        recovered += 1
        X, y = load_dataset(row['dataset'])
        ri, a = np.array(source['outer_reference_indices']), np.array(source['reference_indices'])
        train = np.array(fold['reference_indices'])
        raw, labels = X[ri[a[train]]], y[ri[a[train]]]
        A, _ = preprocess_fit(raw, raw)
        classes = np.array(fit['classes'])
        n, q = len(A), len(classes)
        own = np.searchsorted(classes, labels)
        assert fit['features'] == list(range(A.shape[1])) and fit['edges'] == fold['tree']['edges']
        assert fit['status'] == 'separated' and fit['C'] == fit['pair_penalty'] == 1.
        assert fit['eps'] == .01 and fit['separation_tol'] == 2e-7 and not fit['teacher_guards_enforced']
        if fit_key == 'scaled_fit':
            assert fit['objective_scale'] == n
        else:
            assert fit['epigraph_layout'] == 'rival_only_equivalent'
        W, b = np.array(fit['coef']), np.array(fit['intercept'])
        V = np.array(fit['pair_coef']).reshape(q, len(fit['edges']), 3, 3)
        oracle = forest_box_margins(A, labels, W, b, classes, fit['edges'], V, eps=.01, witnesses=True)
        margins = oracle['pair_margins'].copy()
        margins[np.arange(n), own] = 0.
        np.testing.assert_allclose(margins, fit['envelope_pair_margins'], atol=1e-10)
        losses = logsumexp(-margins, axis=1)
        np.testing.assert_allclose(losses, fit['envelope_losses'], atol=1e-10)
        upper = np.array(fit['upper'])
        np.testing.assert_allclose(upper[np.arange(n), own], 0., atol=1e-7)
        assert np.max(-margins-upper) <= fit['separation_tol']+1e-10
        main = (W[:, fit['features']]@CONTRAST).reshape(q, -1)
        pair = np.einsum('ki,qekl,lj->qeij', CONTRAST, V, CONTRAST).reshape(q, -1)
        beta = np.array(fit['beta'])
        np.testing.assert_allclose(beta, np.column_stack([main, pair]), atol=1e-10)
        np.testing.assert_allclose(beta[0], 0., atol=1e-7)
        np.testing.assert_allclose(b[0], 0., atol=1e-7)
        penalized = beta[1] if q == 2 else beta-beta.mean(axis=0, keepdims=True)
        penalty = np.sum(penalized**2)/(2*n)
        alpha = np.array([1/(q*sum(labels == c)) for c in labels])
        exact = float(alpha@losses+penalty)
        relaxed = float(alpha@logsumexp(upper, axis=1)+penalty)
        np.testing.assert_allclose(exact, fit['exact_envelope_objective'], atol=1e-10)
        np.testing.assert_allclose(relaxed, fit['relaxation_objective'], atol=1e-10)
        assert exact-relaxed <= fit['separation_tol']+1e-8
        for cut in fit['cuts']:
            i, r = cut['object'], cut['rival']
            point = np.array(cut['raw'])
            assert 0 <= i < n and 0 <= r < q and r != own[i]
            lo, hi = np.maximum(0, A[i]-.01), np.minimum(1, A[i]+.01)
            assert np.all(point >= lo-1e-12) and np.all(point <= hi+1e-12)
            vertices = np.stack([lo, hi, np.clip(.5, lo, hi)], axis=1)
            assert np.all(np.min(abs(vertices-point[:, None]), axis=1) < 1e-12)
            score = forest_scores(point[None], W, b, fit['edges'], V)[0]
            assert score[r]-score[own[i]] <= upper[i, r]+fit['separation_tol']+1e-9
            cuts += 1
        for i in sorted(set([0, n//2, n-1])):
            for r in range(q):
                if r == own[i]:
                    continue
                independent = lp_margin(A[i], own[i], r, W, b, fit['edges'], V)
                error = float(abs(independent-margins[i, r]))
                np.testing.assert_allclose(independent, margins[i, r], atol=1e-7)
                probes += 1
                max_error = max(max_error, error)
        pairs += n*(q-1)
        print(path.stem, row['dataset'], row['fold'], 'legacy DP/LP/objective/cuts pass', flush=True)
    assert recovered == saved['recovered'] and len(identities)-recovered == saved['remaining_failed']
    return {'diagnosis': path.stem, 'passed': True, 'recovered': recovered,
            'remaining_failed': len(identities)-recovered, 'all_pair_margins': pairs,
            'cut_replays': cuts, 'independent_LP_probes': probes, 'max_LP_error': max_error,
            'artifact_sha256': digest(path)}


if __name__ == '__main__':
    threadpool_limits(1)
    rows = [audit(ROOT/'results/robust-logloss-scaling-diagnosis.json', 'scaled_fit'),
            audit(ROOT/'results/robust-logloss-rival-epigraph-diagnosis.json', 'rival_fit')]
    result = {'passed': True, 'rows': rows, 'validation_scores_used': False,
              'goal_achieved': False, 'auditor_sha256': digest(Path(__file__)),
              'scope': 'training_numerical_feasibility_not_global_optimality_or_prediction_superiority'}
    (ROOT/'results/robust-logloss-numerical-diagnoses-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result, flush=True)
