"""Independent scalar Lagrangian reconstruction without the production LP."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from diagnose_shared_anchor_enemy_triples import ROOT, FOLDER, PARENT, digest
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.core import interval_bounds
from audit_flexible_majority_repair_admission import independent_protected
from audit_forest_compact_transfer import equivalent


def check_dual(lo, hi, w, check):
    i, f, enemies = check['anchor'], check['friend'], check['enemies']
    p = len(w); left, right = lo[i], hi[i]
    fl = np.maximum(abs(left-lo[f]), abs(left-hi[f]))
    fr = np.maximum(abs(right-lo[f]), abs(right-hi[f]))
    slope = np.divide(fr-fl, right-left, out=np.zeros(p), where=right > left)
    offset = float(w@(fl-slope*left))
    multipliers = np.array(check['multipliers']).reshape(3, 2*p+1)
    assert np.isfinite(multipliers).all() and multipliers.max() <= 0
    rx = np.zeros(p); rz = np.zeros((3, p)); rg = 1.; products = []
    for e, enemy in enumerate(enemies):
        gap = multipliers[e, -1]; rg += gap; rx += gap*w*slope
        products.append(gap*offset)
        for a in range(p):
            pos, neg = multipliers[e, 2*a:2*a+2]
            rx[a] -= pos-neg; rz[e, a] = pos+neg-gap*w[a]
            products.extend([pos*hi[enemy, a], -neg*lo[enemy, a]])
    value = float(sum(products)+np.where(rx >= 0, rx*left, rx*right).sum()+np.minimum(0, rz).sum()-abs(rg)*w.sum())
    equivalent(value, check['raw_lagrangian_bound'])
    padding = 1e-9*(1+abs(value)+sum(abs(v) for v in products)+np.sum(abs(rx)*np.maximum(abs(left), abs(right)))+np.sum(abs(rz))+abs(rg)*w.sum())
    equivalent(value-padding, check['lower_bound'])


def main():
    summary = json.loads((FOLDER/'complete.json').read_text()); counts = Counter(); hashes = {}
    for row in summary['rows']:
        name = row['dataset']; artifact = FOLDER/(name+'.json'); saved = json.loads(artifact.read_text())
        assert saved['complete'] and not saved['query_labels_or_accuracy_used']
        for rel, sha in saved['fingerprints'].items(): assert digest(ROOT/rel) == sha
        parent = json.loads((PARENT/(name+'.json')).read_text())
        X, y = load_dataset(name); ri = np.array(saved['reference_indices'])
        A, _ = preprocess_fit(X[ri], X[ri]); labels = y[ri]; full = np.ones(A.shape[1])/np.sqrt(A.shape[1])
        protected = independent_protected(A, labels, full)
        expected_parents = [r for r in parent['pool'] if len(r['features']) < A.shape[1]]
        assert len(saved['records']) == len(expected_parents)
        additions = 0
        for record, old in zip(saved['records'], expected_parents):
            assert record['features'] == old['features'] and record['origins'] == old['origins']
            fs = record['features']; w = np.array(record['weights']); Z = A[:, fs]; lo, hi = interval_bounds(Z, .01)
            equivalent(w.tolist(), np.array(old['ordinary']['weights'])[fs].tolist())
            nominal = abs(Z[:, None]-Z[None, :])@w; np.fill_diagonal(nominal, np.inf)
            results = {t['anchor']: t for t in record['targets']}; assert len(results) == len(record['targets'])
            correct = np.zeros(len(labels), bool)
            for i in np.flatnonzero(protected):
                friends = np.flatnonzero((labels == labels[i]) & (np.arange(len(labels)) != i))
                friends = friends[np.lexsort((friends, nominal[i, friends]))][:3]
                enemies = np.flatnonzero(labels != labels[i])
                if len(friends) < 3: continue
                def direct(x):
                    near = np.maximum(0, np.maximum(lo[enemies]-x, x-hi[enemies]))
                    far = np.maximum(abs(x-lo[friends]), abs(x-hi[friends]))
                    return near[:, None]-far[None]
                pair = np.minimum(direct(lo[i]), direct(hi[i]))@w
                scalar = pair.min(axis=1); threats = enemies[scalar <= 1e-10*(1+abs(scalar))]
                old_pass = len(threats) <= 2
                if old_pass:
                    assert i not in results; correct[i] = True; continue
                t = results[i]; assert t['friends'] == friends.tolist() and t['threats'] == threats.tolist()
                for check in t['checks']:
                    assert check['anchor'] == i and check['friend'] in friends and set(check['enemies']) <= set(threats)
                    check_dual(lo, hi, w, check); counts['LP_checks'] += 1
                if t['certified']:
                    if t['status'] == 'per_friend_third_enemy_certified':
                        bound = np.partition(pair, 2, axis=0)[2]
                        equivalent(bound.tolist(), t['per_friend_third_bounds'])
                        assert bound.min() > 1e-10*(1+max(abs(bound)))
                    else:
                        assert t['status'] == 'all_enemy_triples_certified'
                        from itertools import combinations
                        lookup = {(tuple(c['enemies']), c['friend']): c for c in t['checks']}
                        for triple in combinations(threats, 3):
                            ids = [int(np.flatnonzero(enemies == e)[0]) for e in triple]
                            for col, friend in enumerate(friends):
                                b = pair[ids, col].max()
                                if b <= 1e-10*(1+abs(b)):
                                    assert lookup[(tuple(triple), int(friend))]['lower_bound'] > 1e-10
                    correct[i] = True
                counts[t['status']] += 1
            fresh = [int(np.sum(correct & protected & (labels == c))) for c in np.unique(labels)]
            assert fresh == record['new_retained_counts']
            assert record['old_retained_counts'] == old['ordinary']['certificate']['retained_counts']
            maj = all(a >= b for a, b in zip(fresh, record['required']))
            dep = old['ordinary']['certificate']['dependency_certificate']['upper_bound'] <= .1 and old['ordinary_class_certificate']['feasible_all_classes']
            assert maj == record['new_majority_feasible'] and dep == record['global_and_class_dependency_feasible']
            assert record['new_joint_feasible'] == (maj and dep)
            counts['nonfull_supports'] += 1
            counts['added_protected_objects'] += sum(fresh)-sum(record['old_retained_counts'])
            added = record['new_joint_feasible'] and not record['old_joint_feasible']
            counts['new_joint_admissions'] += added; additions += added
        assert additions == row['new_joint_admissions']; hashes[name] = digest(artifact)
    equivalent(dict(counts), summary['counts'])
    output = {'passed': True, 'counts': dict(counts), 'artifact_hashes': hashes, 'summary_sha256': digest(FOLDER/'complete.json'),
              'auditor_sha256': digest(Path(__file__)), 'goal_achieved': False,
              'scope': 'independent_scalar_Lagrangian_and_friend_third_order_replay_not_prediction'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(output, indent=2), encoding='utf-8'); print(output, flush=True)


if __name__ == '__main__': main()
