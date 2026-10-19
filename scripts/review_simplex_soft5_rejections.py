"""Audit-gated rejection attribution; compare all mixtures, never query scores."""
import json
from collections import Counter
from run_simplex_soft5_target_admission import ROOT, FOLDER, PARENT, digest
from continue_soft_topk_knn_pilot import verify


def reason(m):
    cert, cls = m['certificate'], m['class_certificate']
    flags = []
    if not cert['majority_feasible']: flags.append('majority')
    if cert['dependency_certificate']['upper_bound'] > .1: flags.append('global_dependency')
    if not cls['feasible_all_classes']: flags.append('classwise_dependency')
    assert bool(flags) == (not m['accepted'])
    return '+'.join(flags) if flags else 'accepted'


def main():
    verify(FOLDER); verify(PARENT)
    counts, rows = Counter(), []
    for name in sorted(verify(FOLDER)['artifact_hashes']):
        current = json.loads((FOLDER/(name+'.json')).read_text())
        parent = json.loads((PARENT/(name+'.json')).read_text())
        local = Counter()
        for r, old in zip(current['records'], parent['records']):
            assert r['features'] == old['features']
            for new, prior in zip(r['mixtures'], old['mixtures']):
                assert new['interpolation'] == prior['interpolation']
                key = reason(new); local['new_'+key] += 1
                local['old_'+reason(prior)] += 1
                if prior['accepted'] and not new['accepted']:
                    local['lost_admission_due_to_'+key] += 1
                if not prior['accepted'] and new['accepted']: local['gained_admission'] += 1
        counts.update(local); rows.append({'dataset': name, 'counts': dict(local)})
    result = {'counts': dict(counts), 'rows': rows, 'query_labels_or_accuracy_used': False,
              'goal_achieved': False, 'scope': 'audited_training_rejection_attribution_not_prediction',
              'audit_sha256': digest(FOLDER/'complete-audit.json')}
    (ROOT/'research/simplex-soft5-rejection-review.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
