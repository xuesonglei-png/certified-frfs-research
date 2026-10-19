"""All86 admitted reference supports; no query labels or classification."""
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from run_soft_topk_target_admission import ROOT, FOLDER as PARENT, digest, GRID
from simplex_soft_topk_targets import fit_simplex_target
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.classwise_dependency_certificate import classwise_dependency_certificate

FOLDER = ROOT/'results/simplex-soft5-target-admission'


def fingerprints(name):
    paths = [Path(__file__), ROOT/'scripts/simplex_soft_topk_targets.py', ROOT/'scripts/soft_topk_predictive_metric.py',
             ROOT/'scripts/capped_predictive_metric.py', ROOT/'scripts/soft_topk_metric_objective.py', ROOT/'research/simplex-soft5-target-admission-protocol.md',
             PARENT/(name+'.json'), PARENT/'complete-audit.json', ROOT/'data/processed'/(name+'.npz')]
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
    records = []
    for old in parent['records']:
        fs, anchor = old['features'], np.array(old['anchor'])
        target, fit = fit_simplex_target(A, labels, fs, anchor)
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
              'rows', fit.get('solver_constraint_rows'), 'admitted', sum(m['accepted'] for m in mixtures), flush=True)
    assert hashes == fingerprints(name)
    saved = {'dataset': name, 'reference_indices': ri.tolist(), 'fingerprints': hashes,
             'records': records, 'complete': True, 'query_labels_or_accuracy_used': False,
             'independent_audit_required': True, 'scope': 'all86_training_anisotropic_soft5_target_admission_not_prediction'}
    out.write_text(json.dumps(saved, separators=(',', ':')), encoding='utf-8')
    return saved


def summarize(reports):
    counts, rows = Counter(), []
    for saved in sorted(reports, key=lambda r: r['dataset']):
        nontrivial = 0; old_total = 0; new_total = 0
        for r in saved['records']:
            fit = r['fit']; counts['supports'] += 1; counts['fit_'+fit['status']] += 1
            counts['optimizer_failures'] += not fit['optimizer_success']
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
               'scope': 'all15_all86_training_only_anisotropic_target_admission_not_accuracy'}
    (FOLDER/'complete.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(summary, flush=True)


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True)
    names = sorted(json.loads((PARENT/'complete-audit.json').read_text())['artifact_hashes'])
    with ProcessPoolExecutor(max_workers=2) as executor:
        reports = [f.result() for f in as_completed([executor.submit(run, name) for name in names])]
    summarize(reports)
