"""All86 admitted reference supports; no query labels or classification."""
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from run_guarded_soft5_descent_admission import ROOT, FOLDER as PARENT, digest, GRID
from seeded_guarded_soft5_descent import fit_seeded_guarded_soft5
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.classwise_dependency_certificate import classwise_dependency_certificate

FOLDER = ROOT/'results/seeded-guarded-soft5-admission'
SEEDS = ROOT/'results/simplex-soft5-target-admission'


def fingerprints(name):
    paths = [Path(__file__), ROOT/'scripts/seeded_guarded_soft5_descent.py', ROOT/'scripts/soft_topk_predictive_metric.py',
             ROOT/'scripts/capped_predictive_metric.py', ROOT/'scripts/soft_topk_metric_objective.py', ROOT/'research/seeded-guarded-soft5-admission-protocol.md',
             PARENT/(name+'.json'), PARENT/'complete-audit.json', SEEDS/(name+'.json'), SEEDS/'complete-audit.json', ROOT/'data/processed'/(name+'.npz')]
    paths += sorted((ROOT/'src/certified_frfs').glob('*.py'))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def run(name):
    threadpool_limits(1)
    parent_audit = json.loads((PARENT/'complete-audit.json').read_text())
    assert parent_audit['passed'] and parent_audit['artifact_hashes'][name] == digest(PARENT/(name+'.json'))
    parent = json.loads((PARENT/(name+'.json')).read_text())
    for rel, sha in parent['fingerprints'].items(): assert digest(ROOT/rel) == sha, rel
    hashes = fingerprints(name); out = FOLDER/(name+'.json')
    if out.exists():
        previous = json.loads(out.read_text()); assert previous['fingerprints'] == hashes
        if previous['complete']: return previous
    X, y = load_dataset(name); ri = np.array(parent['reference_indices']); labels = y[ri]
    A, _ = preprocess_fit(X[ri], X[ri]); full = np.ones(A.shape[1])/np.sqrt(A.shape[1])
    protected = MajorityTerms(A, labels, weights=full).bounds(range(A.shape[1]), allowed_enemies=2) > 1e-10
    seed_audit = json.loads((SEEDS/'complete-audit.json').read_text())
    assert seed_audit['passed'] and seed_audit['artifact_hashes'][name] == digest(SEEDS/(name+'.json'))
    seeds = json.loads((SEEDS/(name+'.json')).read_text())
    assert seeds['reference_indices'] == parent['reference_indices'] and len(seeds['records']) == len(parent['records'])
    for rel, sha in seeds['fingerprints'].items(): assert digest(ROOT/rel) == sha, rel
    records = []
    for old, seed_record in zip(parent['records'], seeds['records']):
        assert old['features'] == seed_record['features']
        np.testing.assert_array_equal(old['anchor'], seed_record['anchor'])
        fs, anchor = old['features'], np.array(old['anchor'])
        target, fit = fit_seeded_guarded_soft5(A, labels, fs, anchor, [seed_record['target']], max_iter=4, max_guard_checks=16)
        mixtures = []
        for t in GRID:
            w = (1-t)*anchor+t*target
            cert = weight_certificate(A, labels, fs, full, w, protected=protected)
            classes = classwise_dependency_certificate(labels, cert)
            mixtures.append({'interpolation': t, 'weights': w.tolist(), 'certificate': cert,
                             'class_certificate': classes, 'accepted': bool(cert['feasible'] and classes['feasible_all_classes']),
                             'weight_shift_l1': float(abs(w-anchor).sum())})
        records.append({'features': fs, 'anchor': anchor.tolist(), 'target': target.tolist(), 'fit': fit,
                        'mixtures': mixtures, 'previous_admitted_mixtures': sum(m['accepted'] for m in old['mixtures']),
                        'previous_endpoint_accepted': old['mixtures'][-1]['accepted']})
        print(name, 'support', len(records), 'size', len(fs), fit['status'],
              'rows', fit['polytope']['constraint_rows'], 'admitted', sum(m['accepted'] for m in mixtures), flush=True)
    assert hashes == fingerprints(name)
    saved = {'dataset': name, 'reference_indices': ri.tolist(), 'fingerprints': hashes,
             'records': records, 'complete': True, 'query_labels_or_accuracy_used': False,
             'independent_audit_required': True, 'scope': 'all86_training_seeded_guarded_soft5_target_admission_not_prediction'}
    out.write_text(json.dumps(saved, separators=(',', ':')), encoding='utf-8')
    return saved


def summarize(reports):
    counts, rows = Counter(), []
    for saved in sorted(reports, key=lambda r: r['dataset']):
        nontrivial = 0; old_total = 0; new_total = 0
        for r in saved['records']:
            fit = r['fit']; counts['supports'] += 1; counts['fit_'+fit['status']] += 1
            counts['selected_improving_seeds'] += fit['selected_seed_id'] is not None; counts['seed_guard_checks'] += len(fit['seed_checks']); counts['guarded_search_checks'] += len(fit['checks']); counts['accepted_search_steps'] += len(fit['accepted_step_ids'])
            counts['previous_admitted_mixtures'] += r['previous_admitted_mixtures']; old_total += r['previous_admitted_mixtures']
            for m in r['mixtures']:
                counts['mixtures'] += 1; counts['admitted_mixtures'] += m['accepted']; new_total += m['accepted']
                if m['accepted'] and m['weight_shift_l1'] > 1e-8:
                    counts['admitted_nontrivial_mixtures'] += 1; nontrivial += 1
            endpoint = r['mixtures'][-1]['accepted']
            counts['new_endpoint_admissions'] += endpoint and not r['previous_endpoint_accepted']
            counts['lost_endpoint_admissions'] += not endpoint and r['previous_endpoint_accepted']
        rows.append({'dataset': saved['dataset'], 'supports': len(saved['records']), 'admitted_nontrivial_mixtures': nontrivial,
                     'previous_admitted_mixtures': old_total, 'admitted_mixtures': new_total})
    summary = {'rows': rows, 'counts': dict(counts), 'goal_achieved': False,
               'query_labels_or_accuracy_used': False, 'independent_audit_required': True,
               'scope': 'all15_all86_training_only_seeded_guarded_pairwise_target_admission_not_accuracy'}
    (FOLDER/'complete.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(summary, flush=True)


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True)
    names = sorted(json.loads((PARENT/'complete-audit.json').read_text())['artifact_hashes'])
    with ProcessPoolExecutor(max_workers=2) as executor:
        reports = [f.result() for f in as_completed([executor.submit(run, name) for name in names])]
    summarize(reports)
