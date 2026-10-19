"""Complete-only review of actual endpoints, retaining adverse comparisons."""
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
from threadpoolctl import threadpool_limits
from run_seeded_guarded_soft5_admission import ROOT, digest
from continue_soft_topk_knn_pilot import verify
from independent_soft_topk_objective import independent_objective
from audit_hardaware_guarded_soft5_admission import independent_hard_BA
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit


def review(family):
    threadpool_limits(1)
    folder = ROOT/'results'/f'{family}-guarded-soft5-admission'
    previous_family = 'guarded-soft5-descent' if family == 'seeded' else 'seeded-guarded-soft5'
    parent = ROOT/'results'/f'{previous_family}-admission'
    audit = verify(folder); verify(parent)
    assert audit['counts']['datasets'] == 15 and audit['counts']['supports'] == 86
    rows, counts, gains, ba_gains = [], Counter(), [], []
    for name in sorted(audit['artifact_hashes']):
        saved = json.loads((folder/(name+'.json')).read_text())
        old = json.loads((parent/(name+'.json')).read_text())
        assert saved['reference_indices'] == old['reference_indices']
        X, y = load_dataset(name); ids = np.array(saved['reference_indices'])
        A, _ = preprocess_fit(X[ids], X[ids]); labels = y[ids]; mass = np.sqrt(A.shape[1])
        targets = []
        for r, previous in zip(saved['records'], old['records'], strict=True):
            assert r['features'] == previous['features']
            np.testing.assert_array_equal(r['anchor'], previous['anchor'])
            fs = r['features']; fit = r['fit']
            anchor_loss = independent_objective(A[:, fs], labels, np.array(r['anchor'])/mass)
            loss = independent_objective(A[:, fs], labels, np.array(r['target'])/mass)
            old_loss = independent_objective(A[:, fs], labels, np.array(previous['target'])/mass)
            np.testing.assert_allclose(loss, fit['training_objective_final'], atol=1e-9, rtol=0)
            anchor_ba = independent_hard_BA(A[:, fs], labels, r['anchor'])
            ba = independent_hard_BA(A[:, fs], labels, r['target'])
            old_ba = independent_hard_BA(A[:, fs], labels, previous['target'])
            counts['supports'] += 1
            counts['status_'+fit['status']] += 1
            counts['seed_checks'] += len(fit['seed_checks'])
            counts['local_guard_checks'] += len(fit['checks'])
            counts['chosen_local_steps'] += len(fit['accepted_step_ids'])
            for label, value in [('soft_loss_vs_parent', old_loss-loss), ('hard_BA_vs_parent', ba-old_ba), ('hard_BA_vs_anchor', ba-anchor_ba)]:
                counts[label+('_improved' if value > 1e-8 else '_worsened' if value < -1e-8 else '_tied')] += 1
            counts['soft_loss_increased_from_anchor'] += loss > anchor_loss+1e-8
            assert fit['endpoint_guard']['accepted']
            counts['certified_endpoints'] += 1
            gains.append(old_loss-loss); ba_gains.append(100*(ba-anchor_ba))
            targets.append({'features': fs, 'status': fit['status'], 'soft_loss': loss,
                            'parent_soft_loss': old_loss, 'soft_loss_decrease_vs_parent': old_loss-loss,
                            'anchor_hard_LOO_BA': anchor_ba, 'target_hard_LOO_BA': ba,
                            'parent_hard_LOO_BA': old_ba,
                            'admitted_mixtures': sum(m['accepted'] for m in r['mixtures']),
                            'seed_checks': len(fit['seed_checks']), 'local_checks': len(fit['checks'])})
        rows.append({'dataset': name, 'targets': targets})
    result = {'family': family, 'rows': rows, 'counts': dict(counts), 'admission_counts': audit['counts'],
              'mean_actual_soft_loss_decrease_vs_parent': float(np.mean(gains)),
              'mean_training_LOO_BA_gain_vs_anchor_pp': float(np.mean(ba_gains)),
              'query_labels_or_accuracy_used': False, 'goal_achieved': False,
              'scope': 'all86_actual_training_endpoints_not_validation_or_test_superiority',
              'audit_sha256': digest(folder/'complete-audit.json'),
              'parent_audit_sha256': digest(parent/'complete-audit.json'),
              'reviewer_sha256': digest(Path(__file__))}
    (ROOT/'research'/f'{family}-guarded-soft5-admission-review.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('family', choices=['seeded', 'hardaware'])
    review(parser.parse_args().family)
