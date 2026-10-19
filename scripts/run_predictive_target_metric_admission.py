"""All admitted multiscale reference supports: predictive target feasibility."""
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_recent_multiscale_pool_admission import ROOT, FOLDER as PARENT, digest
from capped_predictive_metric import fit_predictive_target
from certified_frfs.datasets import load_dataset
from certified_frfs.geometry_only_forest_stage import GeometryStageContext
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.classwise_dependency_certificate import classwise_dependency_certificate

FOLDER = ROOT/'results/predictive-target-metric-admission'
GRID = (.25, .5, .75, 1.)


def run(name):
    threadpool_limits(1)
    audit = json.loads((PARENT/'complete-audit.json').read_text()); source = PARENT/(name+'.json')
    assert audit['passed'] and audit['artifact_hashes'][name] == digest(source)
    parent = json.loads(source.read_text())
    for rel, sha in parent['fingerprints'].items(): assert digest(ROOT/rel) == sha, rel
    files = [Path(__file__), ROOT/'scripts/capped_predictive_metric.py', ROOT/'research/predictive-target-metric-admission-protocol.md',
             source, PARENT/'complete-audit.json', ROOT/'data/processed'/(name+'.npz')]
    files += sorted((ROOT/'src/certified_frfs').glob('*.py'))
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in files}
    out = FOLDER/(name+'.json')
    if out.exists():
        old = json.loads(out.read_text()); assert old['fingerprints'] == hashes
        if old['complete']: return old
    X, y = load_dataset(name); ri = np.array(parent['reference_indices'])
    context = GeometryStageContext(X[ri], y[ri]); records = []
    for previous in parent['pool']:
        if not previous['accepted']: continue
        fs = previous['features']; anchor = np.array(previous['selected_metric']['weights'])[fs]
        target, fit = fit_predictive_target(context.A, y[ri], fs, anchor)
        records_for_grid = []
        for t in GRID:
            weights = (1-t)*anchor+t*target
            cert = weight_certificate(context.A, y[ri], fs, context.original_weights, weights, protected=context.protected)
            classes = classwise_dependency_certificate(y[ri], cert)
            records_for_grid.append({'interpolation': t, 'weights': weights.tolist(), 'certificate': cert,
                                     'class_certificate': classes, 'accepted': cert['feasible'] and classes['feasible_all_classes'],
                                     'weight_shift_l1': float(np.sum(abs(weights-anchor)))})
        records.append({'features': fs, 'origins': previous['origins'], 'anchor': anchor.tolist(), 'target': target.tolist(),
                        'fit': fit, 'mixtures': records_for_grid})
        print(name, 'support', len(records), 'size', len(fs), 'fit', fit['status'], 'admitted mixtures', sum(r['accepted'] for r in records_for_grid), flush=True)
    saved = {'dataset': name, 'reference_indices': ri.tolist(), 'fingerprints': hashes, 'records': records, 'complete': True,
             'query_labels_or_accuracy_used': False, 'independent_audit_required': True,
             'scope': 'all15_training_predictive_target_metric_admission_not_accuracy_or_novelty'}
    assert all(digest(ROOT/rel) == sha for rel, sha in hashes.items())
    out.write_text(json.dumps(saved, separators=(',', ':')), encoding='utf-8'); return saved


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True); names = sorted(json.loads((PARENT/'complete-audit.json').read_text())['artifact_hashes']); reports = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        for future in as_completed([ex.submit(run, name) for name in names]): reports.append(future.result())
    counts = Counter(); rows = []
    for report in sorted(reports, key=lambda r: r['dataset']):
        accepted = 0
        for r in report['records']:
            counts['supports'] += 1; counts['fit_'+r['fit']['status']] += 1
            counts['optimizer_failures'] += not r['fit']['optimizer_success']
            for m in r['mixtures']:
                counts['mixtures'] += 1; counts['admitted_mixtures'] += m['accepted']
                if m['accepted'] and m['weight_shift_l1'] > 1e-8:
                    accepted += 1; counts['admitted_nontrivial_mixtures'] += 1
        rows.append({'dataset': report['dataset'], 'supports': len(report['records']), 'admitted_nontrivial_mixtures': accepted})
    summary = {'rows': rows, 'counts': dict(counts), 'goal_achieved': False, 'query_labels_or_accuracy_used': False,
               'independent_audit_required': True, 'scope': 'training_admission_of_supervised_metric_targets_not_predictive_superiority'}
    (FOLDER/'complete.json').write_text(json.dumps(summary, indent=2), encoding='utf-8'); print(summary, flush=True)
