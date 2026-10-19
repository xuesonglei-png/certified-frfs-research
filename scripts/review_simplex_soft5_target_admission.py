"""Complete-only training review; endpoint loss cannot imply test accuracy."""
import json
from collections import Counter
import numpy as np
from run_simplex_soft5_target_admission import ROOT, FOLDER, PARENT, digest
from continue_soft_topk_knn_pilot import verify
from independent_soft_topk_objective import independent_objective
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit


def main():
    audit = verify(FOLDER); verify(PARENT)
    assert len(audit['artifact_hashes']) == 15 and audit['counts']['supports'] == 86
    rows, counts, losses = [], Counter(), []
    for name in sorted(audit['artifact_hashes']):
        new = json.loads((FOLDER/(name+'.json')).read_text())
        old = json.loads((PARENT/(name+'.json')).read_text())
        X, y = load_dataset(name); ri = np.array(new['reference_indices'])
        A, _ = preprocess_fit(X[ri], X[ri]); labels = y[ri]; d = A.shape[1]
        items = []
        for r, p in zip(new['records'], old['records']):
            assert r['features'] == p['features']
            w = np.array(r['target'])/np.sqrt(d)
            counts['targets_beyond_old_cap'] += w.max() > 2/d+1e-8
            counts['supports'] += 1
            counts['admitted_endpoints_beyond_old_cap'] += r['mixtures'][-1]['accepted'] and w.max() > 2/d+1e-8
            counts['optimizer_failures'] += not r['fit']['optimizer_success']
            counts['explicit_fallbacks'] += 'fallback' in r['fit']['status']
            # A recheck fallback can retain the rejected candidate's objective
            # in optimizer metadata. Compare ACTUAL returned targets instead.
            loss = independent_objective(A[:, r['features']], labels, w)
            previous = independent_objective(A[:, p['features']], labels, np.array(p['target'])/np.sqrt(d))
            gain = previous-loss
            losses.append(gain)
            counts['lower_endpoint_objective'] += gain > 1e-8
            counts['higher_endpoint_objective'] += gain < -1e-8
            items.append({'features': r['features'], 'fit_status': r['fit']['status'],
                          'max_normalized_weight': float(w.max()), 'old_cap': 2/d,
                          'strict_positive_weights': int(sum(w > 0)),
                          'weights_above_1e_minus_8': int(sum(w > 1e-8)),
                          'parent_endpoint_loss_minus_new': gain,
                          'actual_returned_target_objective': loss,
                          'parent_actual_returned_target_objective': previous,
                          'new_endpoint_admitted': r['mixtures'][-1]['accepted'],
                          'parent_endpoint_admitted': p['mixtures'][-1]['accepted'],
                          'new_admitted_mixtures': sum(m['accepted'] for m in r['mixtures']),
                          'parent_admitted_mixtures': sum(m['accepted'] for m in p['mixtures'])})
        rows.append({'dataset': name, 'targets': items})
    assert counts['supports'] == 86
    result = {'counts': {k: int(v) for k, v in counts.items()}, 'admission_counts': audit['counts'], 'rows': rows,
              'mean_parent_endpoint_loss_minus_new': float(np.mean(losses)),
              'scope': 'training_only_local_optimizer_comparison_not_prediction_or_acquisition_savings',
              'objective_source': 'independent_scalar_Brent_actual_returned_targets_including_fallbacks',
              'goal_achieved': False, 'audit_sha256': digest(FOLDER/'complete-audit.json')}
    (ROOT/'research/simplex-soft5-target-admission-review.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__': main()
